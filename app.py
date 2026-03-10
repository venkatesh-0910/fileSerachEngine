"""
Smart PDF Search Engine
========================
Text is extracted at upload time using PyMuPDF (native text).
Scanned/image-only pages are OCR'd in parallel via Tesseract.
Background processing with real-time progress tracking.
Searches are instant via cached JSON.
"""

from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, session, make_response, jsonify,
)
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
from markupsafe import escape
import os
import re
import json
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
ALLOWED_EXTENSIONS = {".pdf"}  # FIX #3: removed .png
SNIPPET_RADIUS = 300
OCR_DPI = 150
MAX_RESULT_STORE = 50  # Max server-side result entries before cleanup

for folder in (UPLOAD_FOLDER, CACHE_FOLDER):
    os.makedirs(folder, exist_ok=True)

# ── Tesseract Configuration ──────────────────────────────────
if os.name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ── Server-side Storage ──────────────────────────────────────
# FIX #1: Store results server-side instead of in session cookie
# FIX #2: Track cancel flags for safe OCR interruption
ocr_tasks = {}          # task_id → {status, done, total, ...}
ocr_cancel_flags = {}   # task_id → bool
result_store = {}       # result_id → {results, keyword}


# ──────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────

def clear_all_files():
    """Remove all uploaded PDFs and cached JSON files."""
    for f in os.listdir(UPLOAD_FOLDER):
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, f))
        except Exception:
            pass
    for f in os.listdir(CACHE_FOLDER):
        try:
            os.remove(os.path.join(CACHE_FOLDER, f))
        except Exception:
            pass


def cancel_running_tasks():
    """Signal all running OCR tasks to stop gracefully."""
    for tid in list(ocr_tasks.keys()):
        if ocr_tasks[tid].get("status") == "running":
            ocr_cancel_flags[tid] = True


def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def no_cache_response(response):
    """Add no-cache headers to prevent browser from caching pages."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


def store_results(results, keyword):
    """
    Store search results server-side with automatic cleanup.
    FIX #1: Avoids ~4KB session cookie size limit.
    """
    # Clean up old entries when store gets too large
    if len(result_store) > MAX_RESULT_STORE:
        keys = list(result_store.keys())
        for k in keys[: len(keys) - MAX_RESULT_STORE // 2]:
            result_store.pop(k, None)

    result_id = str(uuid.uuid4())
    result_store[result_id] = {"results": results, "keyword": keyword}
    return result_id


# ──────────────────────────────────────────────────────────────
# TEXT EXTRACTION (PyMuPDF native text — instant)
# ──────────────────────────────────────────────────────────────

def extract_and_cache(filepath: str, filename: str):
    """
    Open PDF, extract native text from all pages, save to JSON cache.
    Returns (total_pages, indexed_pages, skipped_pages).
    """
    pages = []
    skipped = 0

    with fitz.open(filepath) as doc:
        total_pages = len(doc)
        for i in range(total_pages):
            text = doc.load_page(i).get_text()
            if text.strip():
                pages.append({"page": i + 1, "text": text})
            else:
                skipped += 1

    cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(
            {"filename": filename, "pages": pages},
            f,
            ensure_ascii=False,
        )

    return total_pages, len(pages), skipped


# ──────────────────────────────────────────────────────────────
# PARALLEL OCR FOR SCANNED PAGES
# ──────────────────────────────────────────────────────────────

def ocr_single_page(page_num, img_bytes):
    """Run Tesseract OCR on a single page image."""
    img = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(img, config="--psm 6")
    return page_num, text


def run_ocr_background(filepath, filename, task_id):
    """
    Background worker: extract text from all pages (native + OCR).
    FIX #2: Checks cancellation flag at every stage for safe interruption.
    """
    try:
        # ── Pre-check: bail out if already cancelled or file deleted ──
        if ocr_cancel_flags.get(task_id) or not os.path.exists(filepath):
            ocr_tasks[task_id]["status"] = "cancelled"
            return

        pages = []
        scanned_indices = []

        # ── Phase 1: Quick scan — identify scanned pages ──
        with fitz.open(filepath) as doc:
            total_pages = len(doc)
            for i in range(total_pages):
                text = doc.load_page(i).get_text()
                if text.strip():
                    pages.append({"page": i + 1, "text": text})
                else:
                    scanned_indices.append(i)

        scanned_count = len(scanned_indices)
        ocr_tasks[task_id]["total"] = scanned_count
        ocr_tasks[task_id]["total_pages"] = total_pages

        if scanned_count == 0:
            ocr_tasks[task_id]["status"] = "done"
            ocr_tasks[task_id]["done"] = 0
        else:
            # ── Phase 2: Render + OCR in batches ──
            if ocr_cancel_flags.get(task_id) or not os.path.exists(filepath):
                ocr_tasks[task_id]["status"] = "cancelled"
                return

            max_workers = min(os.cpu_count() or 4, 12)
            batch_size = max_workers * 2
            done_count = 0
            cancelled = False

            with fitz.open(filepath) as doc:
                executor = ThreadPoolExecutor(max_workers=max_workers)
                try:
                    for batch_start in range(0, scanned_count, batch_size):
                        if ocr_cancel_flags.get(task_id):
                            cancelled = True
                            break

                        batch = scanned_indices[
                            batch_start : batch_start + batch_size
                        ]

                        # Render this batch of pages to images
                        batch_data = []
                        for idx in batch:
                            if ocr_cancel_flags.get(task_id):
                                cancelled = True
                                break
                            pix = doc.load_page(idx).get_pixmap(dpi=OCR_DPI)
                            batch_data.append((idx + 1, pix.tobytes("png")))

                        if cancelled:
                            break

                        # OCR this batch in parallel
                        futures = {
                            executor.submit(ocr_single_page, pn, ib): pn
                            for pn, ib in batch_data
                        }
                        for future in as_completed(futures):
                            if ocr_cancel_flags.get(task_id):
                                cancelled = True
                                continue  # Drain remaining futures

                            try:
                                page_num, text = future.result()
                                if text.strip():
                                    pages.append(
                                        {"page": page_num, "text": text}
                                    )
                            except Exception:
                                pass

                            done_count += 1
                            ocr_tasks[task_id]["done"] = done_count

                        del batch_data

                        if cancelled:
                            break

                finally:
                    executor.shutdown(wait=False)

                if cancelled:
                    ocr_tasks[task_id]["status"] = "cancelled"
                    return

        # ── Final check before saving ──
        if ocr_cancel_flags.get(task_id):
            ocr_tasks[task_id]["status"] = "cancelled"
            return

        pages.sort(key=lambda x: x["page"])

        cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {"filename": filename, "pages": pages},
                f,
                ensure_ascii=False,
            )

        ocr_tasks[task_id]["status"] = "done"
        ocr_tasks[task_id]["indexed"] = len(pages)

    except Exception as e:
        # File deleted mid-process → treat as cancellation, not error
        if not os.path.exists(filepath):
            ocr_tasks[task_id]["status"] = "cancelled"
            return
        ocr_tasks[task_id]["status"] = "error"
        ocr_tasks[task_id]["error"] = str(e)


# ──────────────────────────────────────────────────────────────
# SNIPPETS
# ──────────────────────────────────────────────────────────────

def build_snippet(text, keyword, radius=SNIPPET_RADIUS):
    """Extract a snippet around the keyword with highlighting."""
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

    # FIX #6: Preserve original case in highlighted matches
    highlighted = re.sub(
        re.escape(escaped_kw),
        lambda m: f"<mark>{m.group()}</mark>",
        safe_snippet,
        flags=re.IGNORECASE,
    )
    return prefix + highlighted.replace("\n", "<br>") + suffix


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────

# FIX #8: Handle file size limit exceeded
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("File is too large. Maximum upload size is 50 MB.", "danger")
    return redirect(url_for("index"))


@app.route("/")
def index():
    """
    Home page.
    FIX #1: Results retrieved from server-side store, not session.
    FIX #5: No longer auto-clears files on every visit.
    """
    # Retrieve server-side results (if any)
    result_id = session.pop("result_id", None)
    stored = result_store.pop(result_id, {}) if result_id else {}
    results = stored.get("results")
    keyword = stored.get("keyword")

    uploaded_file = session.pop("uploaded_file", None)
    active_task_id = session.pop("active_task_id", None)

    # If no uploaded_file in session, check disk for existing files
    if uploaded_file:
        uploaded_files = [uploaded_file]
    else:
        uploaded_files = [
            f
            for f in os.listdir(UPLOAD_FOLDER)
            if f.lower().endswith(".pdf")
        ]

    cached_files = [
        f.replace(".json", "")
        for f in os.listdir(CACHE_FOLDER)
        if f.endswith(".json")
    ]

    # Recover active OCR task on page refresh (session was popped)
    if not active_task_id:
        for tid, task in ocr_tasks.items():
            if task.get("status") == "running":
                active_task_id = tid
                break

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
    """Handle PDF upload with safe OCR cancellation on re-upload."""
    file = request.files.get("file")

    if not file or file.filename == "":
        flash("Please select a file to upload.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Only PDF files are allowed.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    if not filename:
        flash("Invalid filename.", "danger")
        return redirect(url_for("index"))

    # FIX #2: Cancel any running OCR tasks before clearing files
    cancel_running_tasks()
    clear_all_files()

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        # Quick scan: check if any pages are scanned (image-only)
        has_scanned = False
        with fitz.open(filepath) as doc:
            for i in range(len(doc)):
                if not doc.load_page(i).get_text().strip():
                    has_scanned = True
                    break

        if has_scanned:
            # Start background OCR with parallel processing
            task_id = str(uuid.uuid4())
            ocr_tasks[task_id] = {
                "status": "running",
                "done": 0,
                "total": 0,
                "total_pages": 0,
                "filename": filename,
                "error": None,
            }
            ocr_cancel_flags[task_id] = False

            thread = threading.Thread(
                target=run_ocr_background,
                args=(filepath, filename, task_id),
                daemon=True,
            )
            thread.start()

            flash(
                f"'{filename}' uploaded! OCR processing started — "
                f"scanned pages are being processed in parallel...",
                "success",
            )
            session["uploaded_file"] = filename
            session["active_task_id"] = task_id
        else:
            # All pages have native text — extract instantly
            total_pages, indexed, skipped = extract_and_cache(
                filepath, filename
            )
            flash(
                f"'{filename}' uploaded and indexed successfully! "
                f"All {total_pages} pages indexed.",
                "success",
            )
            session["uploaded_file"] = filename

    except Exception as e:
        flash(f"Upload OK but processing failed: {e}", "danger")

    return redirect(url_for("index"))


@app.route("/progress/<task_id>")
def progress(task_id):
    """Return OCR progress as JSON for the frontend progress bar."""
    task = ocr_tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"})
    return jsonify(
        {
            "status": task["status"],
            "done": task["done"],
            "total": task["total"],
            "error": task.get("error"),
        }
    )


@app.route("/search", methods=["POST"])
def search():
    """Search cached JSON files for keyword matches."""
    keyword = request.form.get("keyword", "").strip()
    results = []

    if not keyword:
        flash("Please enter a keyword to search.", "warning")
        uploaded_files = [
            f
            for f in os.listdir(UPLOAD_FOLDER)
            if f.lower().endswith(".pdf")
        ]
        if uploaded_files:
            session["uploaded_file"] = uploaded_files[0]
        return redirect(url_for("index"))

    keyword_lower = keyword.lower()

    cache_files = [
        f for f in os.listdir(CACHE_FOLDER) if f.endswith(".json")
    ]
    if not cache_files:
        flash(
            "No PDF uploaded yet. Please upload a PDF first.", "warning"
        )
        return redirect(url_for("index"))

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
                results.append(
                    {
                        "filename": data["filename"],
                        "page": page_info["page"],
                        "text": snippet,
                    }
                )

    if not results:
        flash(f"No results found for '{keyword}'.", "info")

    # FIX #1: Store results server-side (avoids ~4KB session cookie limit)
    result_id = store_results(results, keyword)
    session["result_id"] = result_id

    if current_uploaded:
        session["uploaded_file"] = current_uploaded

    return redirect(url_for("index"))


@app.route("/clear", methods=["POST"])
def clear():
    """Delete all uploaded PDFs, cached data, and cancel OCR tasks."""
    cancel_running_tasks()
    clear_all_files()
    ocr_tasks.clear()
    ocr_cancel_flags.clear()
    result_store.clear()
    session.clear()
    flash("All files cleared.", "info")
    return redirect(url_for("index"))


# FIX #7: Guard startup cleanup & disable reloader to prevent double-run
if __name__ == "__main__":
    clear_all_files()
    app.run(debug=True, use_reloader=False)