"""
Smart PDF Search Engine (Optimized)
====================================
Text is extracted at upload time.
- Plain text pages: extracted instantly in a single pass (no threading overhead)
- Scanned pages: parallel OCR with ThreadPoolExecutor in a background thread
Searches are instant via cached JSON.
Every page refresh starts completely fresh.
"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify
from werkzeug.utils import secure_filename
from markupsafe import escape
import os
import re
import json
import uuid
import threading
import time
import fitz  # PyMuPDF
import pytesseract
from PIL import Image, ImageFilter
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
ALLOWED_EXTENSIONS = {".pdf", ".png"}
SNIPPET_RADIUS = 300

# ── Tuning knobs ──────────────────────────────────────────────
OCR_DPI = 100                       # Lower = faster render, still accurate
OCR_MAX_WORKERS = min(os.cpu_count() or 4, 8)   # Use available CPU cores
MAX_OCR_PAGES = 2000
TESSERACT_CONFIG = (
    "--oem 1 "
    "--psm 6 "
    "-l eng"
)
# ──────────────────────────────────────────────────────────────

for folder in (UPLOAD_FOLDER, CACHE_FOLDER):
    os.makedirs(folder, exist_ok=True)

pytesseract.pytesseract.tesseract_cmd = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# ──────────────────────────────────────────────────────────────
# BACKGROUND OCR TASK STATE
# ──────────────────────────────────────────────────────────────
ocr_tasks = {}          # task_id → {status, done, total, filename, error}
ocr_tasks_lock = threading.Lock()


def clear_all_files():
    """Remove all uploaded PDFs and cached JSON files."""
    for f in os.listdir(UPLOAD_FOLDER):
        filepath = os.path.join(UPLOAD_FOLDER, f)
        try:
            os.remove(filepath)
        except Exception:
            pass
    for f in os.listdir(CACHE_FOLDER):
        filepath = os.path.join(CACHE_FOLDER, f)
        try:
            os.remove(filepath)
        except Exception:
            pass


# Clean up on startup
clear_all_files()


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def no_cache_response(response):
    """Add no-cache headers to prevent browser from caching pages."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# ──────────────────────────────────────────────────────────────
# OCR HELPERS (only used for scanned pages)
# ──────────────────────────────────────────────────────────────

def _ocr_from_png_bytes(png_data: bytes) -> str:
    """
    Run Tesseract on pre-rendered PNG bytes.
    Called inside a worker thread — no file I/O needed.
    """
    img = Image.open(io.BytesIO(png_data))
    img = img.convert("L")
    img = img.point(lambda x: 0 if x < 180 else 255)
    text = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
    return text


def _ocr_page_from_bytes(page_number: int, png_data: bytes) -> dict:
    """
    OCR a single page from pre-rendered PNG bytes.
    No file access needed — runs entirely in memory.
    """
    text = _ocr_from_png_bytes(png_data)
    return {"page": page_number + 1, "text": text}


# ──────────────────────────────────────────────────────────────
# TEXT EXTRACTION + PAGE PRE-RENDERING (single file open)
# ──────────────────────────────────────────────────────────────

def extract_text_pages(filepath: str, prerender_ocr=False):
    """
    Open PDF ONCE, extract all native text pages instantly.
    If prerender_ocr=True, also render scanned pages to PNG bytes
    so worker threads never need to reopen the PDF.

    Returns: (pages_dict, ocr_needed_list, total_pages, rendered_images)
             rendered_images = {page_number: png_bytes} (only if prerender_ocr)
    """
    pages = {}
    ocr_needed = []
    rendered_images = {}

    zoom = OCR_DPI / 72
    mat = fitz.Matrix(zoom, zoom)

    with fitz.open(filepath) as doc:
        total_pages = len(doc)
        for i in range(total_pages):
            page = doc.load_page(i)
            text = page.get_text()

            if text.strip():
                pages[i] = {"page": i + 1, "text": text}
            else:
                ocr_needed.append(i)
                if prerender_ocr and len(ocr_needed) <= MAX_OCR_PAGES:
                    # Render to PNG bytes while file is open
                    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
                    rendered_images[i] = pix.tobytes("png")
                    del pix  # Free memory immediately

    return pages, ocr_needed, total_pages, rendered_images


# ──────────────────────────────────────────────────────────────
# BACKGROUND OCR (Pass 2: threaded, with progress tracking)
# ──────────────────────────────────────────────────────────────

def run_background_ocr(task_id: str, filepath: str, filename: str,
                       pages: dict, ocr_needed: list,
                       rendered_images: dict):
    """
    Run OCR on pre-rendered page images in a background thread.
    Workers receive PNG bytes directly — no file I/O needed.
    Updates ocr_tasks[task_id] with progress.
    """
    try:
        # Limit OCR pages
        if len(ocr_needed) > MAX_OCR_PAGES:
            ocr_needed_trimmed = ocr_needed[:MAX_OCR_PAGES]
            print(f"Warning: Limiting OCR to first {MAX_OCR_PAGES} of "
                  f"{len(ocr_needed)} scanned pages")
        else:
            ocr_needed_trimmed = ocr_needed

        total_ocr = len(ocr_needed_trimmed)

        with ocr_tasks_lock:
            ocr_tasks[task_id]["total"] = total_ocr
            ocr_tasks[task_id]["status"] = "processing"

        print(f"[Task {task_id}] OCR starting: {total_ocr} pages, "
              f"{OCR_MAX_WORKERS} workers, {OCR_DPI} DPI")

        with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
            futures = {
                pool.submit(_ocr_page_from_bytes, i, rendered_images[i]): i
                for i in ocr_needed_trimmed
                if i in rendered_images
            }
            done_count = 0
            for future in as_completed(futures):
                page_num = futures[future]
                try:
                    pages[page_num] = future.result()
                except Exception as e:
                    pages[page_num] = {"page": page_num + 1, "text": ""}
                    print(f"OCR error on page {page_num + 1}: {e}")

                done_count += 1
                with ocr_tasks_lock:
                    ocr_tasks[task_id]["done"] = done_count

                # Free rendered image memory after processing
                rendered_images.pop(page_num, None)

        # Sort by page number and save cache
        sorted_pages = [pages[i] for i in sorted(pages.keys())]
        cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"filename": filename, "pages": sorted_pages},
                f,
                ensure_ascii=False,
            )

        with ocr_tasks_lock:
            ocr_tasks[task_id]["status"] = "done"

        print(f"[Task {task_id}] OCR complete — {total_ocr} pages processed")

    except Exception as e:
        with ocr_tasks_lock:
            ocr_tasks[task_id]["status"] = "error"
            ocr_tasks[task_id]["error"] = str(e)
        print(f"[Task {task_id}] OCR failed: {e}")


def extract_and_cache(filepath: str, filename: str) -> None:
    """
    Legacy synchronous extraction for text-only PDFs.
    Called when no OCR is needed.
    """
    pages, ocr_needed, total_pages = extract_text_pages(filepath)

    if not ocr_needed:
        print(f"All {total_pages} pages have native text — no OCR needed")
        sorted_pages = [pages[i] for i in sorted(pages.keys())]
        cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"filename": filename, "pages": sorted_pages},
                f,
                ensure_ascii=False,
            )


# ──────────────────────────────────────────────────────────────
# SNIPPETS
# ──────────────────────────────────────────────────────────────

def build_snippet(text, keyword, radius=SNIPPET_RADIUS):
    """
    Extract a snippet of text around the keyword match,
    with radius characters of context on each side.
    """
    match = re.search(re.escape(keyword), text, re.IGNORECASE)
    if not match:
        return ""

    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    snippet = text[start:end]

    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""

    safe_snippet = str(escape(snippet))
    escaped_kw = str(escape(keyword))
    highlighted = re.sub(
        re.escape(escaped_kw),
        f"<mark>{escaped_kw}</mark>",
        safe_snippet,
        flags=re.IGNORECASE,
    )
    return prefix + highlighted.replace("\n", "<br>") + suffix


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """
    Home page.
    Always clears everything on a fresh GET request.
    Only shows data when redirected from upload/search via POST-redirect.
    """
    results = session.pop("results", None)
    keyword = session.pop("keyword", None)
    uploaded_file = session.pop("uploaded_file", None)
    active_task_id = session.pop("active_task_id", None)

    if not uploaded_file:
        clear_all_files()
        uploaded_files = []
        cached_files = []
    else:
        uploaded_files = [uploaded_file]
        cached_files = [uploaded_file]

    response = make_response(
        render_template(
            "index.html",
            uploaded_files=uploaded_files,
            cached_files=cached_files,
            results=results,
            keyword=keyword,
            active_task_id=active_task_id,
        )
    )
    return no_cache_response(response)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle PDF upload, extract text, start background OCR if needed."""
    file = request.files.get("file")

    if not file or file.filename == "":
        flash("Please select a file to upload.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are allowed.", "danger")
        return redirect(url_for("index"))

    clear_all_files()

    # Clear any old OCR tasks
    with ocr_tasks_lock:
        ocr_tasks.clear()

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Pass 1: Extract native text (instant)
        pages, ocr_needed, total_pages = extract_text_pages(filepath)

        if ocr_needed:
            # Start background OCR
            task_id = str(uuid.uuid4())[:8]
            with ocr_tasks_lock:
                ocr_tasks[task_id] = {
                    "status": "starting",
                    "done": 0,
                    "total": len(ocr_needed),
                    "filename": filename,
                    "error": None,
                }

            thread = threading.Thread(
                target=run_background_ocr,
                args=(task_id, filepath, filename, pages, ocr_needed),
                daemon=True,
            )
            thread.start()

            # Save text pages immediately so partial search works
            if pages:
                sorted_pages = [pages[i] for i in sorted(pages.keys())]
                cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"filename": filename, "pages": sorted_pages},
                        f,
                        ensure_ascii=False,
                    )

            session["uploaded_file"] = filename
            session["active_task_id"] = task_id

            flash(
                f"'{filename}' uploaded! {total_pages - len(ocr_needed)} text pages indexed. "
                f"OCR running on {len(ocr_needed)} scanned pages…",
                "success",
            )
        else:
            # All text — save cache immediately (no OCR needed)
            extract_and_cache(filepath, filename)
            flash(f"'{filename}' uploaded and indexed successfully! All {total_pages} pages have native text.", "success")
            session["uploaded_file"] = filename

    except Exception as e:
        flash(f"Upload OK but text extraction failed: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/progress/<task_id>")
def progress(task_id):
    """Return current OCR progress for a background task."""
    with ocr_tasks_lock:
        task = ocr_tasks.get(task_id)

    if not task:
        return jsonify({"status": "not_found"}), 404

    return jsonify({
        "status": task["status"],
        "done": task["done"],
        "total": task["total"],
        "filename": task["filename"],
        "error": task["error"],
    })


@app.route("/search", methods=["POST"])
def search():
    """Search cached JSON files for keyword matches."""
    keyword = request.form.get("keyword", "").strip()
    results = []

    if not keyword:
        flash("Please enter a keyword to search.", "warning")
        uploaded_files = [
            f for f in os.listdir(UPLOAD_FOLDER) if f.lower().endswith(".pdf")
        ]
        if uploaded_files:
            session["uploaded_file"] = uploaded_files[0]
        return redirect(url_for("index"))

    keyword_lower = keyword.lower()

    cache_files = [f for f in os.listdir(CACHE_FOLDER) if f.endswith(".json")]
    if not cache_files:
        flash("No PDF uploaded yet. Please upload a PDF first.", "warning")
        return redirect(url_for("index"))

    # Check if OCR is still running
    any_running = False
    with ocr_tasks_lock:
        for task in ocr_tasks.values():
            if task["status"] in ("starting", "processing"):
                any_running = True
                break

    if any_running:
        flash(
            "⚠️ OCR is still processing scanned pages. "
            "Search results may be incomplete — try again after processing finishes.",
            "warning",
        )

    current_uploaded = None

    for cache_file in cache_files:
        cache_path = os.path.join(CACHE_FOLDER, cache_file)
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        current_uploaded = data["filename"]

        for page_info in data["pages"]:
            text = page_info["text"]
            if keyword_lower in text.lower():
                snippet = build_snippet(text, keyword)
                results.append({
                    "filename": data["filename"],
                    "page": page_info["page"],
                    "text": snippet,
                })

    if not results:
        flash(f"No results found for '{keyword}'.", "info")

    session["results"] = results
    session["keyword"] = keyword
    if current_uploaded:
        session["uploaded_file"] = current_uploaded

    # Preserve active task if OCR is still running
    if any_running:
        with ocr_tasks_lock:
            for tid, task in ocr_tasks.items():
                if task["status"] in ("starting", "processing"):
                    session["active_task_id"] = tid
                    break

    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear():
    """Delete all uploaded PDFs and cached data."""
    clear_all_files()
    with ocr_tasks_lock:
        ocr_tasks.clear()
    session.clear()
    flash("All files cleared.", "info")
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)