# 📄 Smart PDF Search Engine v2

A powerful, locally-hosted document intelligence and search application. This tool goes beyond basic text matching by integrating **Optical Character Recognition (OCR)**, **AI-Powered Semantic Search**, and **Intelligent Topic Detection** to help you extract insights from any PDF document—whether it's natively digital or a scanned image.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)
![Machine Learning](https://img.shields.io/badge/AI-Sentence--Transformers-FF6F00)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Key Features

### 🔍 Advanced Search Capabilities
- **Exact Keyword Search**: Instantly find specific terms across all pages with context-aware highlighted snippets.
- **🧠 Semantic Search (v2)**: Powered by `sentence-transformers` (`all-MiniLM-L6-v2`), search queries are embedded to find philosophically and contextually similar topics, even if the exact keyword is missing from the document.

### 📑 Intelligent Document Analysis
- **Topic & Chapter Detection**: Automatically analyzes document structures by examining font sizes and weights via PyMuPDF to deduce architectural headings.
- **TF-IDF Fallback**: If structural headings aren't found, the app uses Term Frequency-Inverse Document Frequency (TF-IDF) to analyze page chunks and generate summarized topic labels.

### 👁️ Robust OCR & PDF Parsing
- **Lightning Native Extraction**: Instantly parses digital PDFs using PyMuPDF.
- **Parallel Scanned PDF Processing**: Detects image-only pages and deploys Tesseract OCR across multiple CPU threads simultaneously for rapid text extraction.

### ⚡ Performance & UX
- **Real-Time Progress Tracking**: Heavy OCR and AI embedding generation happens in background threads, mapped to live UI progress bars—you're never left waiting blindly.
- **Smart Local Caching**: Extracted text (`.json`), detected topics (`.topics.json`), and generated embeddings (`.npz`) are cached to disk. Subsequent searches and page loads are nearly instantaneous.
- **Graceful Interruptions**: Re-uploading files instantly and safely cancels any background OCR or embedding tasks.
- **Server-Side Result Storage**: Avoids memory limitations and cookie bloat by securely storing search sessions on the backend.
- **Dark & Light Themes**: Accessible, modern UI built with Bootstrap 5 and smooth transitions.
- **100% Private**: Everything (including machine learning models) runs locally on your machine. Zero data leaves your computer.

---

## 🛠️ Technology Stack

| Architecture Layer | Technologies Used |
|--------------------|-------------------|
| **Backend & Routing** | Flask, Werkzeug |
| **PDF Processing** | PyMuPDF (fitz) |
| **Optical Character Recognition** | Tesseract OCR, pytesseract, Pillow |
| **Machine Learning & NLP** | `sentence-transformers`, `scikit-learn` (TF-IDF, Cosine Similarity), `numpy` |
| **Generative Pipeline** | ThreadPoolExecutor (Parallel processing) |
| **Frontend** | HTML5, CSS3, Vanilla JS, Bootstrap 5, Google Fonts (Inter) |

---

## 📁 Project Architecture

```text
project/
├── app.py              # Application core, routing, background workers, AI logic
├── templates/
│   └── index.html      # Responsive single-page UI (Search, Results, Progress)
├── uploads/            # Temporary storage for uploaded PDFs (git-ignored)
├── cache/              # Cached text extractions and topic summaries (git-ignored)
├── embeddings/         # Cached sentence-transformer .npz files (git-ignored)
├── venv/               # Python virtual environment
├── .gitignore
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1. System Prerequisites

- **Python 3.8+**
- **Tesseract OCR** installed on your system.

#### Installing Tesseract
- **Windows**: Download the installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) and install to `C:\Program Files\Tesseract-OCR\`. (Ensure the path is correct as it is hardcoded in `app.py`).
- **macOS**: `brew install tesseract`
- **Linux (Debian/Ubuntu)**: `sudo apt install tesseract-ocr`

### 2. Installation

Clone the repository and navigate into the directory:
```bash
git clone <your-repo-url>
cd project
```

Create and activate a virtual environment:
```bash
# Verify Python version first
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

Install the required Python dependencies:
```bash
pip install flask PyMuPDF pytesseract Pillow markupsafe werkzeug sentence-transformers scikit-learn numpy
```
*(Note: Running the app for the first time will automatically download the ~90MB `all-MiniLM-L6-v2` embedding model).*

### 3. Running the Application

Start the Flask development server:
```bash
python app.py
```

Access the application in your browser:
```text
http://127.0.0.1:5000
```

---

## 📖 Usage Guide

1. **Upload**: Drag and drop or browse to upload a `.pdf` file (Up to 50MB by default).
2. **Processing Pipeline**: 
    - The engine first checks for native text.
    - If empty, the multithreaded OCR engine processes the document, showing a live progress bar.
    - An AI Background Worker then begins Topic Detection and Semantic Embedding generation.
3. **Search**: 
    - Enter a search query or keyword.
    - If found exactly, you'll receive highlighted contextual snippets.
    - The system will also perform a **Semantic Search**, retrieving pages and overarching topics contextually related to your query through AI vector matching.
4. **Clean Up**: Use the **Clear All Files** button to purge the `uploads`, `cache`, and `embeddings` directories.

---

## ⚙️ Core Configuration

Configuration settings can be tweaked inside `app.py`:

| Setting Variable | Default | Description |
|------------------|---------|-------------|
| `MAX_CONTENT_LENGTH` | 50 MB | Enforces maximum upload file size limit. |
| `SNIPPET_RADIUS` | 300 chars | Defines context length around highlighted keyword snippets. |
| `OCR_DPI` | 150 | Determines the render resolution for scanned PDF pages. |
| `MAX_RESULT_STORE` | 50 | Purges oldest server-side results to prevent session bloat. |
| `SEMANTIC_THRESHOLD` | 0.25 | Minimum Cosine Similarity score for a semantic match to surface. |
| `MAX_SEMANTIC_RESULTS` | 8 | The maximum number of semantic results to display per search. |

---

## 📊 System Diagrams

### 1. Entity-Relation Diagram (Data Structures)
```mermaid
erDiagram
    UPLOADED-PDF ||--o{ CACHED-JSON : generates
    UPLOADED-PDF ||--o{ CACHED-TOPICS : analyzed_into
    UPLOADED-PDF ||--o{ EMBEDDINGS-NPZ : analyzed_into
    CACHED-JSON }|--|| PAGES : contains
    CACHED-TOPICS }|--|| TOPICS : contains
    RESULT-STORE ||--o{ SEARCH-RESULTS : stores
    OCR-TASKS ||--|| PDF-PROCESSING : tracks
    ANALYSIS-TASKS ||--|| AI-PROCESSING : tracks
```

### 2. Data Flow Diagram (DFD)
```mermaid
flowchart TD
    User([User]) -->|Uploads PDF| FlaskApp(Flask App)
    FlaskApp -->|Checks pages| PyMuPDF{PyMuPDF Native Text?}
    PyMuPDF -->|Yes| CacheJSON[(Cache JSON)]
    PyMuPDF -->|No (Scanned)| OCRWorker[OCR Worker Thread]
    OCRWorker -->|Extracts Text| CacheJSON
    
    CacheJSON --> AIWorker[AI Analysis Worker]
    AIWorker -->|Font/TF-IDF Analysis| CacheTopics[(Topics JSON)]
    AIWorker -->|sentence-transformers| CacheEmbeddings[(Embeddings NPZ)]
    
    User -->|Enters Query| SearchLogic(Search Engine)
    SearchLogic -->|Reads| CacheJSON
    SearchLogic -->|Reads| CacheTopics
    SearchLogic -->|Reads| CacheEmbeddings
    SearchLogic -->|Saves| ResultStore[(Result Store UUID)]
    ResultStore -->|Returns Display| UI(Display UI)
    UI --> User
```

### 3. Module / Class Diagram
```mermaid
classDiagram
    class FlaskApp {
        +upload()
        +index()
    }
    class DocumentProcessor {
        +extract_and_cache()
        +ocr_single_page()
        +run_ocr_background()
    }
    class AIAnalyzer {
        +detect_topics()
        +generate_embeddings()
        +run_analysis_background()
    }
    class SearchEngine {
        +semantic_search()
        +build_snippet()
        +store_results()
    }
    class LocalState {
        -ocr_tasks: dict
        -result_store: dict
        -analysis_tasks: dict
    }
    
    FlaskApp --> DocumentProcessor : triggers
    FlaskApp --> SearchEngine : queries
    DocumentProcessor --> AIAnalyzer : triggers
    DocumentProcessor --> LocalState : updates
    AIAnalyzer --> LocalState : updates
    SearchEngine --> LocalState : reads
```

### 4. Use Case Diagram
```mermaid
flowchart LR
    User([User]) --> U1(Upload PDF)
    User --> U2(Search Document)
    User --> U3(Clear Data)
    U1 -.->|Triggers| U4(OCR Processing)
    U1 -.->|Triggers| U5(AI Topic Detection)
    U2 -.->|Includes| U6(Semantic Match)
    U2 -.->|Includes| U7(Snippet Highlighting)
```

### 5. Sequence Diagram
```mermaid
sequenceDiagram
    actor User
    participant Frontend
    participant Backend
    participant PyMuPDF
    participant TesseractWorker
    participant AIWorker
    participant DiskCache
    
    User->>Frontend: Upload PDF
    Frontend->>Backend: POST /upload
    Backend->>PyMuPDF: Scan PDF type
    alt Has Native Text
        PyMuPDF-->>Backend: Native text
        Backend->>DiskCache: Save JSON
    else Is Scanned / Images
        PyMuPDF-->>Backend: Scanned images detected
        Backend->>TesseractWorker: Start parallel background OCR
        TesseractWorker->>DiskCache: Save JSON when done
    end
    Backend->>AIWorker: Kickoff AI Analysis
    Backend-->>Frontend: 200 OK (Task Started)
    Frontend->>Backend: Poll /status (ProgressBar)
    AIWorker->>DiskCache: Save Topics (.json) & Embeddings (.npz)
    
    User->>Frontend: Submit Search Keyword
    Frontend->>Backend: POST /search
    Backend->>DiskCache: Retrieve JSON & NPZ
    Backend->>Backend: Compute Exact Matches & Cosine Similarities
    Backend->>DiskCache: Store Result Set UUID
    Backend-->>Frontend: Display Search Results & Snippets
```

### 6. Activity Diagram
```mermaid
stateDiagram-v2
    [*] --> Upload_PDF
    Upload_PDF --> Check_Native_Text
    Check_Native_Text --> Extract_Instantly : Text Found
    Check_Native_Text --> Parallel_OCR : Image-Only
    
    Parallel_OCR --> Save_JSON_Cache
    Extract_Instantly --> Save_JSON_Cache
    
    Save_JSON_Cache --> AI_Topic_Formatting
    AI_Topic_Formatting --> NLP_Sentence_Embeddings
    NLP_Sentence_Embeddings --> Save_NPZ_Embeddings
    
    state Search_Process {
        [*] --> Enter_Query
        Enter_Query --> Literal_Keyword_Match
        Enter_Query --> Semantic_Cosine_Similarity
        Literal_Keyword_Match --> Aggregate_Results
        Semantic_Cosine_Similarity --> Aggregate_Results
        Aggregate_Results --> Store_Result_UUID
        Store_Result_UUID --> Render_UI
    }
    
    Save_NPZ_Embeddings --> Search_Process
```

### 7. File Upload & Validation Flowchart
```mermaid
flowchart TD
    Start(User Uploads File) --> ValidExt{Is .pdf?}
    ValidExt -- No --> UIError(Flash Error: Invalid File)
    ValidExt -- Yes --> ValidSize{< 50MB?}
    ValidSize -- No --> UILarge(Flash Error: Too Large)
    ValidSize -- Yes --> Cancel[Cancel running OCR/AI tasks]
    Cancel --> CleanCache[Clear previous uploads/cache]
    CleanCache --> Save[Save to /uploads]
    Save --> Next[Trigger Processing Pipeline]
```

### 8. AI Topic Detection Flowchart
```mermaid
flowchart TD
    Start[AI Worker Starts] --> LoadJSON[Load Cached JSON Text]
    LoadJSON --> Strategy1[Strategy 1: Font Size Analysis via PyMuPDF]
    Strategy1 --> Found{Found > 2 Headings?}
    Found -- Yes --> Format1[Format Headings & Summarize]
    Found -- No --> Strategy2[Strategy 2: TF-IDF Fallback]
    Strategy2 --> ChunkPages[Group pages into chunks of 5]
    ChunkPages --> TFIDF[Run TF-IDF Vectorizer]
    TFIDF --> TopWords[Extract Top 3 keywords per chunk]
    TopWords --> Format2[Format as Sections & Summarize]
    Format1 --> Save[(Save topics.json)]
    Format2 --> Save
```

### 9. Semantic Search Ranking Flowchart
```mermaid
flowchart TD
    Start[User Searches Keyword] --> CheckNPZ{Embeddings .npz Exists?}
    CheckNPZ -- No --> ExactMatch(Return only exact keyword matches)
    CheckNPZ -- Yes --> TransformQuery[Model encodes keyword into vector]
    TransformQuery --> LoadNPZ[Load page embeddings]
    LoadNPZ --> CosineSim[Compute Scikit-learn Cosine Similarity]
    CosineSim --> Threshold{Score >= 0.25?}
    Threshold -- No --> ExactMatch
    Threshold -- Yes --> TopK[Sort by highest score, take Top 8]
    TopK --> JoinText[Fetch page text & topic summaries]
    JoinText --> Highlight[Generate context snippet]
    Highlight --> Return(Return Semantic Results to UI)
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---
> Engineered with ❤️ using Flask, PyMuPDF, sentence-transformers, and Tesseract OCR.
