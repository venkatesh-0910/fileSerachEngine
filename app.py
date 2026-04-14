# ─────────────────────────────────────────────────────
# app.py  –  Smart PDF Search Engine (backend)
#
# Quick guide to find stuff (Ctrl+F these):
#   "PART 1"  → imports
#   "PART 2"  → AI model loader
#   "PART 3"  → flask app setup
#   "PART 4"  → folders & constants
#   "PART 5"  → tesseract config
#   "PART 6"  → global storage dicts
#   "PART 7"  → helper / utility functions
#   "PART 8"  → topic detection logic
#   "PART 9"  → embedding generation
#   "PART 10" → background analysis worker
#   "PART 11" → semantic search
#   "PART 12" → text extraction (PyMuPDF)
#   "PART 13" → OCR processing (Tesseract)
#   "PART 14" → snippet builder
#   "PART 15" → all Flask routes
#   "PART 16" → main entry point
# ─────────────────────────────────────────────────────

"""
Smart PDF Search Engine
========================
Text is extracted at upload time using PyMuPDF (native text).
Scanned/image-only pages are OCR'd in parallel via Tesseract.
Background processing with real-time progress tracking.
Searches are instant via cached JSON.

Topic Analysis & Semantic Search (v2):
- Detects chapters/sections via font-size analysis (PyMuPDF)
- Falls back to TF-IDF keyword clustering for topic detection
- Generates sentence embeddings (all-MiniLM-L6-v2) for semantic search
- When exact keyword search fails, returns related topics via cosine similarity
"""

# ── PART 1: Imports ──────────────────────────────────
# all the libraries we need for the project
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
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# ── PART 2: AI Model Loader ──────────────────────────
# loads the sentence-transformer model (downloads ~90MB first time)
# using lazy loading so it only loads when we actually need it
_st_model = None
_st_lock = threading.Lock()

def get_st_model():
    """Lazy-load and cache the sentence-transformer model."""
    global _st_model
    if _st_model is None:
        with _st_lock:
            if _st_model is None:
                from sentence_transformers import SentenceTransformer
                _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model

# ── PART 3: Flask App Setup ──────────────────────────
# creating the flask app and setting the secret key + max upload size
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB upload limit

# ── PART 4: Folders & Constants ──────────────────────
# setting up directories and some config values
# like what file types are allowed, OCR resolution, etc.
UPLOAD_FOLDER = "uploads"
CACHE_FOLDER = "cache"
EMBEDDINGS_FOLDER = "embeddings"
ALLOWED_EXTENSIONS = {".pdf"}  # FIX #3: removed .png
SNIPPET_RADIUS = 300
OCR_DPI = 150
MAX_RESULT_STORE = 50  # Max server-side result entries before cleanup
SEMANTIC_THRESHOLD = 0.25  # Minimum cosine similarity for semantic results
MAX_SEMANTIC_RESULTS = 8  # Max semantic matches to return

for folder in (UPLOAD_FOLDER, CACHE_FOLDER, EMBEDDINGS_FOLDER):
    os.makedirs(folder, exist_ok=True)

# ── PART 5: Tesseract OCR Config ─────────────────────
# telling python where tesseract is installed (windows path)
if os.name == "nt":
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path

# ── PART 6: Global Storage (dictionaries) ────────────
# these dicts keep track of running tasks, results, etc.
# we store results here instead of in cookies because
# cookies have a ~4KB size limit and our results can be bigger
ocr_tasks = {}          # task_id → {status, done, total, ...}
ocr_cancel_flags = {}   # task_id → bool
result_store = {}       # result_id → {results, keyword}
analysis_tasks = {}     # task_id → {status, filename, topics, ...}
analysis_cancel_flags = {} # task_id → bool


# ── PART 7: Helper / Utility Functions ───────────────
# small reusable functions:
#   clear_all_files()    → deletes everything (uploads, cache, embeddings)
#   cancel_running_tasks() → stops any OCR or analysis running in background
#   allowed_file()       → checks if uploaded file is a .pdf
#   no_cache_response()  → tells browser not to cache the page
#   store_results()      → saves search results on server side

def clear_all_files():
    """Remove all uploaded PDFs, cached JSON, and embedding files."""
    for folder in (UPLOAD_FOLDER, CACHE_FOLDER, EMBEDDINGS_FOLDER):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except Exception:
                pass


def cancel_running_tasks():
    """Signal all running tasks to stop gracefully."""
    for tid in list(ocr_tasks.keys()):
        if ocr_tasks[tid].get("status") == "running":
            ocr_cancel_flags[tid] = True
            
    for tid in list(analysis_tasks.keys()):
        if analysis_tasks[tid].get("status") in ["running", "pending"]:
            analysis_cancel_flags[tid] = True


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


# ── PART 8: Topic Detection ──────────────────────────
# tries to figure out what topics/chapters are in the PDF
# method 1: looks at font sizes (bigger font = heading = topic)
# method 2: if that doesn't work, uses TF-IDF to find keywords
# then adds a short summary from the page text for each topic

def detect_topics(filepath: str, cached_pages: list):
    """
    Detect topics/sections from a PDF using two strategies:
    1. Font-size analysis: larger fonts → headings → topic names
    2. TF-IDF fallback: extract top keywords per page group as topics
    Returns list of {title, page, summary, text_chunk}.
    """
    topics = []

    # Strategy 1: Heading detection via font size
    try:
        with fitz.open(filepath) as doc:
            # Collect all font sizes to determine the body-text baseline
            all_sizes = []
            for i in range(min(len(doc), 30)):  # Sample first 30 pages
                blocks = doc.load_page(i).get_text("dict", flags=0)["blocks"]
                for b in blocks:
                    for line in b.get("lines", []):
                        for span in line.get("spans", []):
                            sz = round(span["size"], 1)
                            txt = span["text"].strip()
                            if txt and len(txt) > 1:
                                all_sizes.append(sz)

            if all_sizes:
                # Body text is usually the most common font size
                from collections import Counter
                size_counts = Counter(all_sizes)
                body_size = size_counts.most_common(1)[0][0]
                heading_threshold = body_size + 1.5  # Headings are bigger

                for i in range(len(doc)):
                    page = doc.load_page(i)
                    blocks = page.get_text("dict", flags=0)["blocks"]
                    for b in blocks:
                        for line in b.get("lines", []):
                            for span in line.get("spans", []):
                                sz = round(span["size"], 1)
                                txt = span["text"].strip()
                                is_bold = "bold" in span.get("font", "").lower()
                                if txt and len(txt) > 3 and len(txt) < 150:
                                    if sz >= heading_threshold or (
                                        sz >= body_size + 0.5 and is_bold
                                    ):
                                        topics.append({
                                            "title": txt,
                                            "page": i + 1,
                                            "font_size": sz,
                                        })
    except Exception:
        pass

    # Deduplicate and clean topics
    seen_titles = set()
    clean_topics = []
    for t in topics:
        normalized = t["title"].lower().strip()
        # Skip page numbers, very short titles, or duplicates
        if (
            normalized not in seen_titles
            and len(normalized) > 3
            and not normalized.replace(".", "").replace(" ", "").isdigit()
        ):
            seen_titles.add(normalized)
            clean_topics.append(t)

    # Strategy 2: TF-IDF fallback if few headings found
    if len(clean_topics) < 3 and len(cached_pages) > 2:
        try:
            # Group pages into chunks of 5
            chunks = []
            chunk_pages = []
            for i in range(0, len(cached_pages), 5):
                group = cached_pages[i : i + 5]
                combined = " ".join(p["text"] for p in group)
                chunks.append(combined)
                chunk_pages.append(group[0]["page"])

            if chunks:
                vectorizer = TfidfVectorizer(
                    max_features=100, stop_words="english",
                    min_df=1, max_df=0.9
                )
                tfidf = vectorizer.fit_transform(chunks)
                feature_names = vectorizer.get_feature_names_out()

                for idx, row in enumerate(tfidf.toarray()):
                    top_indices = row.argsort()[-3:][::-1]
                    top_words = [feature_names[j] for j in top_indices if row[j] > 0]
                    if top_words:
                        title = " / ".join(w.title() for w in top_words)
                        normalized = title.lower()
                        if normalized not in seen_titles:
                            seen_titles.add(normalized)
                            clean_topics.append({
                                "title": f"Section: {title}",
                                "page": chunk_pages[idx],
                                "font_size": 0,
                            })
        except Exception:
            pass

    # Enrich topics with summaries from the page text
    for topic in clean_topics:
        page_num = topic["page"]
        # Find the page text
        page_text = ""
        for p in cached_pages:
            if p["page"] == page_num:
                page_text = p["text"]
                break

        # Extract first 2-3 sentences as summary
        if page_text:
            sentences = re.split(r'(?<=[.!?])\s+', page_text.strip())
            summary_sentences = [s for s in sentences[:5] if len(s) > 20][:3]
            topic["summary"] = " ".join(summary_sentences)[:400]
            # Store a text chunk for embedding
            topic["text_chunk"] = page_text[:1000]
        else:
            topic["summary"] = ""
            topic["text_chunk"] = topic["title"]

    # Sort by page number
    clean_topics.sort(key=lambda x: x["page"])
    return clean_topics


# ── PART 9: Embedding Generation ─────────────────────
# converts page text & topics into number vectors (embeddings)
# so we can do similarity search later
# saves them as .npz files so we don't have to redo this every time
def generate_embeddings(filename: str, cached_pages: list, topics: list):
    """
    Generate embeddings for all pages and topics using sentence-transformers.
    Saves to embeddings/<filename>.npz
    """
    model = get_st_model()

    # Create text chunks for each page
    page_texts = []
    page_nums = []
    for p in cached_pages:
        text = p["text"][:1500]  # Limit chunk size
        if text.strip():
            page_texts.append(text)
            page_nums.append(p["page"])

    # Create text chunks for each topic
    topic_texts = [t.get("text_chunk", t["title"]) for t in topics]

    # Encode everything
    all_texts = page_texts + topic_texts
    if not all_texts:
        return

    embeddings = model.encode(all_texts, show_progress_bar=False)

    page_embeddings = embeddings[: len(page_texts)]
    topic_embeddings = embeddings[len(page_texts) :]

    # Save to disk
    emb_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.npz")
    np.savez(
        emb_path,
        page_embeddings=page_embeddings,
        page_nums=np.array(page_nums),
        topic_embeddings=topic_embeddings,
    )


# ── PART 10: Background Analysis Worker ──────────────
# this runs in a separate thread so the user doesn't have to wait
# it does topic detection + embedding generation in the background
# can be cancelled if user uploads a new file
def run_analysis_background(filepath, filename, task_id):
    """Background worker: detect topics + generate embeddings."""
    try:
        if analysis_cancel_flags.get(task_id):
            if task_id in analysis_tasks:
                analysis_tasks[task_id]["status"] = "cancelled"
            return

        analysis_tasks[task_id]["status"] = "running"

        # Wait for cache file to be ready
        cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
        for _ in range(3600):  # Wait up to 60 minutes
            if analysis_cancel_flags.get(task_id):
                if task_id in analysis_tasks:
                    analysis_tasks[task_id]["status"] = "cancelled"
                return
            if os.path.exists(cache_path):
                break
            import time
            time.sleep(1)

        if not os.path.exists(cache_path):
            if task_id in analysis_tasks:
                analysis_tasks[task_id]["status"] = "error"
                analysis_tasks[task_id]["error"] = "Cache file not ready"
            return
            
        if analysis_cancel_flags.get(task_id):
            if task_id in analysis_tasks:
                analysis_tasks[task_id]["status"] = "cancelled"
            return

        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cached_pages = data.get("pages", [])

        # Step 1: Detect topics
        analysis_tasks[task_id]["step"] = "Detecting topics..."
        topics = detect_topics(filepath, cached_pages)
        analysis_tasks[task_id]["topics"] = topics

        # Step 2: Generate embeddings
        if task_id in analysis_tasks:
            analysis_tasks[task_id]["step"] = "Generating embeddings..."
            
        if analysis_cancel_flags.get(task_id):
            if task_id in analysis_tasks:
                analysis_tasks[task_id]["status"] = "cancelled"
            return
            
        generate_embeddings(filename, cached_pages, topics)

        # Save topics to a JSON file
        topics_path = os.path.join(CACHE_FOLDER, f"{filename}.topics.json")
        # Clean topics for JSON serialization (remove text_chunk)
        saveable = [
            {"title": t["title"], "page": t["page"], "summary": t["summary"]}
            for t in topics
        ]
        
        if analysis_cancel_flags.get(task_id):
            if task_id in analysis_tasks:
                analysis_tasks[task_id]["status"] = "cancelled"
            return
            
        with open(topics_path, "w", encoding="utf-8") as f:
            json.dump(saveable, f, ensure_ascii=False)

        if task_id in analysis_tasks:
            analysis_tasks[task_id]["status"] = "done"
            analysis_tasks[task_id]["topic_count"] = len(topics)

    except Exception as e:
        if task_id in analysis_tasks:
            analysis_tasks[task_id]["status"] = "error"
            analysis_tasks[task_id]["error"] = str(e)
        else:
            analysis_tasks[task_id] = {"status": "error", "error": str(e)}


# ── PART 11: Semantic Search ─────────────────────────
# when normal keyword search finds nothing, this kicks in
# it converts the search query into a vector and compares it
# against all the page embeddings using cosine similarity
# returns pages that are "similar" even if they don't have the exact word
def semantic_search(keyword: str, filename: str):
    """
    Perform semantic search using cosine similarity against stored embeddings.
    Returns list of {filename, page, text, score, topic_title, summary}.
    """
    emb_path = os.path.join(EMBEDDINGS_FOLDER, f"{filename}.npz")
    if not os.path.exists(emb_path):
        return []

    cache_path = os.path.join(CACHE_FOLDER, f"{filename}.json")
    topics_path = os.path.join(CACHE_FOLDER, f"{filename}.topics.json")

    if not os.path.exists(cache_path):
        return []

    try:
        model = get_st_model()
        query_embedding = model.encode([keyword])

        data = np.load(emb_path)
        page_embeddings = data["page_embeddings"]
        page_nums = data["page_nums"]

        # Cosine similarity against all page embeddings
        similarities = sklearn_cosine(query_embedding, page_embeddings)[0]

        # Load page texts
        with open(cache_path, "r", encoding="utf-8") as f:
            cache_data = json.load(f)

        # Load topics
        topics = []
        if os.path.exists(topics_path):
            with open(topics_path, "r", encoding="utf-8") as f:
                topics = json.load(f)

        # Find pages with similarity above threshold
        results = []
        scored = list(zip(page_nums, similarities))
        scored.sort(key=lambda x: x[1], reverse=True)

        for page_num, score in scored[:MAX_SEMANTIC_RESULTS]:
            if score < SEMANTIC_THRESHOLD:
                break

            # Find the page text
            page_text = ""
            for p in cache_data.get("pages", []):
                if p["page"] == int(page_num):
                    page_text = p["text"]
                    break

            # Find the topic for this page
            topic_title = ""
            topic_summary = ""
            for t in topics:
                if t["page"] <= int(page_num):
                    topic_title = t["title"]
                    topic_summary = t.get("summary", "")

            # Build a snippet (first 400 chars of page text)
            snippet = page_text[:400].replace("\n", " ").strip()
            if len(page_text) > 400:
                snippet += "..."

            results.append({
                "filename": cache_data.get("filename", filename),
                "page": int(page_num),
                "text": snippet,
                "score": round(float(score), 3),
                "topic_title": topic_title,
                "topic_summary": topic_summary,
            })

        return results

    except Exception:
        return []


# ── PART 12: Text Extraction (fast method) ───────────
# uses PyMuPDF to grab all the text from the PDF
# this is super fast for PDFs that already have selectable text
# saves everything to a JSON file so we don't have to extract again

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


# ── PART 13: OCR Processing (for scanned pages) ─────
# some PDFs are basically just images (scanned documents)
# for those, we use Tesseract OCR to read the text from images
# runs in parallel threads to speed things up
# can be cancelled at any point if user wants to stop

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


# ── PART 14: Snippet Builder ─────────────────────────
# when we find a keyword match, we don't show the entire page
# instead we cut out a small snippet around the match
# and wrap the keyword in <mark> tags so it shows up highlighted

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


# ── PART 15: Flask Routes (all the URL endpoints) ────
# this is where all the pages and API endpoints are defined

# handles the error when someone tries to upload a file bigger than 50MB
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    flash("File is too large. Maximum upload size is 50 MB.", "danger")
    return redirect(url_for("index"))


# home page - shows the upload form, search bar, and results
@app.route("/")
def index():
    """
    Home page.
    FIX #1: Results retrieved from server-side store, not session.
    FIX #5: No longer auto-clears files on every visit.
    """
    # Retrieve server-side results (if any)
    result_id = session.get("result_id", None)
    stored = result_store.get(result_id, {}) if result_id else {}
    results = stored.get("results")
    keyword = stored.get("keyword")
    semantic_results = stored.get("semantic_results")

    uploaded_file = session.get("uploaded_file", None)
    active_task_id = session.get("active_task_id", None)
    analysis_task_id = session.get("analysis_task_id", None)

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
        if f.endswith(".json") and not f.endswith(".topics.json")
    ]

    response = make_response(
        render_template(
            "index.html",
            uploaded_files=uploaded_files,
            cached_files=cached_files,
            results=results,
            keyword=keyword,
            active_task_id=active_task_id,
            analysis_task_id=analysis_task_id,
            semantic_results=semantic_results,
        )
    )
    return no_cache_response(response)


# handles file upload - saves the PDF and starts processing it
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
                "start_time": time.time(),
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

        # Start AI topic analysis in the background
        analysis_id = str(uuid.uuid4())
        analysis_tasks[analysis_id] = {
            "status": "pending",
            "filename": filename,
            "step": "Queued...",
            "topics": [],
            "topic_count": 0,
            "error": None,
        }
        analysis_cancel_flags[analysis_id] = False
        analysis_thread = threading.Thread(
            target=run_analysis_background,
            args=(filepath, filename, analysis_id),
            daemon=True,
        )
        analysis_thread.start()
        session["analysis_task_id"] = analysis_id

    except Exception as e:
        flash(f"Upload OK but processing failed: {e}", "danger")

    return redirect(url_for("index"))


# returns OCR progress as JSON (the frontend polls this every second)
@app.route("/progress/<task_id>")
def progress(task_id):
    """Return OCR progress as JSON for the frontend progress bar."""
    task = ocr_tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"})
    elapsed = time.time() - task.get("start_time", time.time())
    return jsonify(
        {
            "status": task["status"],
            "done": task["done"],
            "total": task["total"],
            "error": task.get("error"),
            "elapsed": round(elapsed, 1),
        }
    )


# search route - looks for keyword in cached text, falls back to AI search
@app.route("/search", methods=["POST"])
def search():
    """Search cached JSON files for keyword matches, with semantic fallback."""
    keyword = request.form.get("keyword", "").strip()
    results = []
    semantic_results = []

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
        f for f in os.listdir(CACHE_FOLDER)
        if f.endswith(".json") and not f.endswith(".topics.json")
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

    # Semantic search fallback when no exact matches found
    if not results and current_uploaded:
        try:
            semantic_results = semantic_search(keyword, current_uploaded)
            if semantic_results:
                flash(
                    f"No exact match for '{keyword}', but found "
                    f"{len(semantic_results)} related topic(s) using AI analysis.",
                    "info",
                )
            else:
                flash(f"No results found for '{keyword}'.", "info")
        except Exception:
            flash(f"No results found for '{keyword}'.", "info")
    elif not results:
        flash(f"No results found for '{keyword}'.", "info")

    # Store results server-side (avoids ~4KB session cookie limit)
    result_id = store_results(results, keyword)
    result_store[result_id]["semantic_results"] = semantic_results
    session["result_id"] = result_id

    if current_uploaded:
        session["uploaded_file"] = current_uploaded

    return redirect(url_for("index"))


# returns topic analysis progress (frontend polls this every 2 seconds)
@app.route("/analysis-progress/<task_id>")
def analysis_progress(task_id):
    """Return analysis progress as JSON for the frontend."""
    task = analysis_tasks.get(task_id)
    if not task:
        return jsonify({"status": "not_found"})
    return jsonify({
        "status": task["status"],
        "step": task.get("step", ""),
        "topic_count": task.get("topic_count", 0),
        "error": task.get("error"),
    })


# returns the detected topics for a PDF as JSON
@app.route("/topics/<filename>")
def get_topics(filename):
    """Return detected topics for a given PDF as JSON."""
    safe_filename = secure_filename(filename)
    topics_path = os.path.join(CACHE_FOLDER, f"{safe_filename}.topics.json")
    if not os.path.exists(topics_path):
        return jsonify({"topics": [], "status": "not_found"})
    with open(topics_path, "r", encoding="utf-8") as f:
        topics = json.load(f)
    return jsonify({"topics": topics, "status": "ok"})


# clears everything - deletes files, stops tasks, resets session
@app.route("/clear", methods=["POST"])
def clear():
    """Delete all uploaded PDFs, cached data, and cancel OCR tasks."""
    cancel_running_tasks()
    clear_all_files()
    ocr_tasks.clear()
    ocr_cancel_flags.clear()
    result_store.clear()
    analysis_tasks.clear()
    analysis_cancel_flags.clear()
    session.clear()
    flash("All files cleared.", "info")
    return redirect(url_for("index"))


# ── PART 16: Main Entry Point ────────────────────────
# starts the flask server
# clears old files on startup so we get a fresh start
# use_reloader=False because it was causing the app to run twice
if __name__ == "__main__":
    clear_all_files()
    app.run(debug=True, use_reloader=False)