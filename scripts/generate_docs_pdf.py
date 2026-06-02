"""
Generate NeuralVault_Documentation.pdf — a complete technical document
covering the architecture, the vector-DB optimization story, every build
phase, how the parts fit together, the API, testing, and deployment.

Run:
    python scripts/generate_docs_pdf.py
"""
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted, Table, TableStyle,
    PageBreak, ListFlowable, ListItem,
)

OUT = "NeuralVault_Documentation.pdf"

# --------------------------------------------------------------------------- #
# Styles                                                                      #
# --------------------------------------------------------------------------- #
styles = getSampleStyleSheet()

INK     = colors.HexColor("#1c222a")
ACCENT  = colors.HexColor("#b87d28")
MUTED   = colors.HexColor("#5b6675")
CODEBG  = colors.HexColor("#f4f1ea")
CODEBORDER = colors.HexColor("#d9d2c4")
RULE    = colors.HexColor("#c9cfd8")

title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=30,
                             textColor=INK, leading=36, spaceAfter=8)
subtitle_style = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13,
                                textColor=MUTED, alignment=TA_CENTER, leading=18)
h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=18, textColor=INK,
                    spaceBefore=18, spaceAfter=8, leading=22)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13.5, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=5, leading=17)
h3 = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=11.5, textColor=INK,
                    spaceBefore=8, spaceAfter=3, leading=15)
body = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, textColor=INK,
                      leading=15, spaceAfter=6, alignment=TA_LEFT)
code_style = ParagraphStyle("Code", parent=styles["Code"], fontSize=8, leading=10.5,
                            textColor=INK, backColor=CODEBG, borderColor=CODEBORDER,
                            borderWidth=0.5, borderPadding=6, spaceBefore=4, spaceAfter=8)
toc_style = ParagraphStyle("TOC", parent=body, leading=18, spaceAfter=2)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Builders                                                                    #
# --------------------------------------------------------------------------- #
S = []  # story


def P(text):     S.append(Paragraph(text, body))
def H1(text):    S.append(Paragraph(text, h1))
def H2(text):    S.append(Paragraph(text, h2))
def H3(text):    S.append(Paragraph(text, h3))
def gap(h=6):    S.append(Spacer(1, h))
def br():        S.append(PageBreak())
def code(text):  S.append(Preformatted(esc(text.strip("\n")), code_style))


def bullets(items):
    S.append(ListFlowable(
        [ListItem(Paragraph(t, body), leftIndent=6) for t in items],
        bulletType="bullet", start="•", leftIndent=14, bulletColor=ACCENT,
    ))
    gap(2)


def table(rows, widths, header=True):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    if header:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("BACKGROUND", (0, 0), (-1, 0), INK),
        ]
    t.setStyle(TableStyle(style))
    S.append(t)
    gap(8)


# --------------------------------------------------------------------------- #
# Page furniture (footer + page numbers)                                      #
# --------------------------------------------------------------------------- #
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.1 * cm, "NeuralVault — Technical Documentation")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(2 * cm, 1.4 * cm, A4[0] - 2 * cm, 1.4 * cm)
    canvas.restoreState()


# =========================================================================== #
# CONTENT
# =========================================================================== #

# ---- Cover ----
S.append(Spacer(1, 5 * cm))
S.append(Paragraph("NeuralVault", title_style))
S.append(Paragraph("Semantic Long-Term Memory Engine for LLM Systems", subtitle_style))
gap(6)
S.append(Paragraph("Complete Technical Documentation", subtitle_style))
S.append(Spacer(1, 1.2 * cm))
S.append(Paragraph(f"Generated {date.today().isoformat()}", subtitle_style))
S.append(Paragraph("github.com/GeorgeAyman1/NeuralVault", subtitle_style))
br()

# ---- TOC ----
H1("Contents")
toc = [
    "1.  Executive Summary",
    "2.  What NeuralVault Is",
    "3.  The Two-Model Architecture",
    "4.  System Architecture",
    "5.  Repository Layout",
    "6.  The Vector Database — Optimization Story",
    "7.  Storage Layer",
    "8.  Sprint 0 — Wiring the Engine In",
    "9.  Sprint 1 — Memory Intelligence",
    "10. Sprint 2 — Document & Codebase Ingestion",
    "11. Sprint 3 — LLM Integration",
    "12. Sprint 4 — Production Hardening",
    "13. CI/CD, Docker, and Frontend",
    "14. End-to-End: A Chat Request",
    "15. API Reference",
    "16. Configuration Reference",
    "17. Testing",
    "18. Running NeuralVault",
    "19. Engineering Decisions & Lessons",
    "20. Glossary",
]
for t in toc:
    S.append(Paragraph(t, toc_style))
br()

# ---- 1. Executive Summary ----
H1("1. Executive Summary")
P("NeuralVault is a semantic long-term memory layer for large language models. "
  "Where an LLM's context window forgets everything between sessions, NeuralVault "
  "stores knowledge as vector embeddings, retrieves the most relevant memories for "
  "any query in milliseconds, and feeds them to the model as grounded context.")
P("The system was built in five phases on top of a hand-optimized vector database. "
  "The vector database is the foundation: it searches <b>20 million</b> 64-dimensional "
  "vectors in roughly <b>0.04 seconds</b> with near-zero per-query RAM, using an "
  "IVF (inverted-file) index. On top of that sit memory-management intelligence, "
  "multi-format ingestion, an LLM answering layer with prompt caching, production "
  "monitoring, a CI pipeline, a Docker deployment, and a demo web UI.")
P("The codebase ships with <b>100 automated tests</b> (all passing, fully offline) "
  "and a GitHub Actions pipeline that runs lint and tests on every push.")

H2("Headline numbers")
table([
    ["Metric", "Value"],
    ["Retrieval @ 20M vectors", "~0.04 s/query (was 2.58 s brute force)"],
    ["Per-query RAM @ 20M", "~0 MB (was 2,333 MB before tuning)"],
    ["Retrieval accuracy (score)", "0.0 — perfect (every result in the true top set)"],
    ["Embedding dimension", "384 (memory app) / 64 (benchmark DB)"],
    ["Automated tests", "100, passing, offline"],
    ["Build phases", "Sprint 0–4 + CI/CD + frontend"],
], [6.5 * cm, 9.5 * cm])

# ---- 2. What it is ----
H1("2. What NeuralVault Is")
P("Traditional databases search for exact matches. NeuralVault searches for "
  "<b>meaning</b>. Instead of storing only text, it stores vector embeddings that "
  "capture the semantic relationships between memories. A query like "
  "\"when is the deadline?\" retrieves a memory that says \"the project is due June 15\" "
  "even though they share no keywords, because their embeddings point in a similar "
  "direction in vector space.")
H2("What it does")
bullets([
    "<b>Stores</b> memories as normalized embeddings plus metadata (text, UUID, timestamp, importance, retrieval count).",
    "<b>Retrieves</b> the top-k most semantically similar memories for any query.",
    "<b>Manages</b> memory over time: merge duplicates, prune stale entries, summarize clusters.",
    "<b>Ingests</b> knowledge from notebooks, PDFs, DOCX files, and whole code directories.",
    "<b>Answers</b> questions by grounding an LLM (Claude Opus 4.8) in the retrieved memories.",
    "<b>Serves</b> all of this through a FastAPI REST API, a demo web UI, and a Docker container.",
])

# ---- 3. Two-model architecture ----
H1("3. The Two-Model Architecture")
P("A common point of confusion: NeuralVault uses <b>two completely different models</b> "
  "for two different jobs. Understanding this split is the key to understanding the system.")
table([
    ["", "Embedding model", "Generation model"],
    ["Name", "all-MiniLM-L6-v2", "claude-opus-4-8"],
    ["Where", "core/embeddings/encoder.py", "core/llm/llm_client.py"],
    ["Job", "Turn text into 384-dim vectors", "Reason over memories, write the answer"],
    ["Runs", "Locally (CPU), free", "Anthropic API, paid"],
    ["Needs key?", "No", "Yes (ANTHROPIC_API_KEY)"],
], [2.6 * cm, 6.7 * cm, 6.7 * cm])
P("The embedding model decides <i>which</i> memories are relevant; the generation "
  "model decides <i>what to say</i> about them. This is why <b>/memory/search</b> works "
  "with no API key (embeddings only) while <b>/memory/chat</b> returns HTTP 503 without "
  "one (it needs Claude). Either model can be swapped without touching the other.")

# ---- 4. System architecture ----
H1("4. System Architecture")
P("Data flows top-to-bottom on write, and bottom-to-top on read:")
code(
"""        [ User / LLM / Document ]
                  |
        +---------v----------+
        |  Ingestion layer   |  chunking, notebook/pdf/docx/dir loaders
        +---------+----------+
                  |
        +---------v----------+
        |   TextEncoder      |  all-MiniLM-L6-v2  -> 384-dim vector
        +---------+----------+
                  |
   +--------------v---------------+
   |        MemoryService         |  orchestrator (the hub)
   +---+-------------+--------+----+
       |             |        |
 +-----v----+  +-----v----+  +v-----------+
 | VecDBStore|  | Metadata |  | LLM layer  |
 | (vectors) |  |  Store   |  | (chat)     |
 +-----+-----+  +----------+  +-----+------+
       |                            |
 +-----v-----+               +------v------+
 |  VecDB    |               | ContextBuilder
 | IVF index |               | Conversation
 +-----------+               | LLMClient -> Claude
""")
P("Every component is reachable through <b>MemoryService</b>, which is the single "
  "orchestration hub. The FastAPI routes are thin wrappers that call its methods.")

# ---- 5. Repo layout ----
H1("5. Repository Layout")
code(
"""NeuralVault/
  vec_db.py                  # the optimized vector database (VecDB class)
  main.py                    # FastAPI app: middleware, health, /ui mount
  core/
    embeddings/encoder.py    # TextEncoder (all-MiniLM-L6-v2)
    storage/
      vecdb_store.py         # VecDBStore  (brute-force / IVF, soft-delete)
      metadata_store.py      # MetadataStore (JSON: text, id, timestamps)
    indexing/
      retrieval.py           # SemanticRetriever
      hybrid_retrieval.py    # semantic + keyword reranking
      keyword_search.py
    memory/
      service.py             # MemoryService  (the hub)
      consolidation.py       # duplicate detection
      pruning.py             # stale-memory scoring
      summarization.py       # extractive cluster summary
      decay.py, ranking.py   # recency/importance/frequency scoring
    ingestion/
      chunking.py, notebook_loader.py,
      pdf_loader.py, docx_loader.py, directory_loader.py
    llm/
      context_builder.py     # cached-prefix system prompt
      conversation.py        # multi-turn history
      llm_client.py          # Anthropic SDK wrapper
    utils/
      config.py, metrics.py, logging.py
  api/routes/                # memory.py, health.py
  api/schemas/               # pydantic request models
  scripts/                   # build_index.py, inspect_db.py, ...
  frontend/index.html        # demo UI
  tests/                     # 100 tests + conftest.py
  Dockerfile, docker-compose.yml, .github/workflows/test.yml
""")

# ---- 6. The optimization story ----
br()
H1("6. The Vector Database — Optimization Story")
P("The heart of NeuralVault is <b>vec_db.py</b>. Its job sounds simple: given a query "
  "vector, find the most similar stored vectors by cosine similarity. The challenge is "
  "doing it fast and cheap at 20 million vectors. This is the story of how it got there.")

H2("6.1 Brute force — the baseline")
P("The naive approach compares the query against every stored vector. Correct, but it "
  "scales linearly. Measured on a 20M x 64 database:")
table([
    ["DB size", "Brute-force time"],
    ["1M", "0.21 s"],
    ["10M", "0.91 s"],
    ["20M", "2.58 s  (too slow)"],
], [5 * cm, 6 * cm])

H2("6.2 Three correctness fixes first")
P("Before speed, three real bugs had to be fixed — fast but wrong is worthless:")
bullets([
    "<b>.dat loading:</b> raw binary files were memory-mapped without a shape, producing "
    "a flat 1-D array so the matrix multiply silently broke. Fixed to mmap with an "
    "explicit (N, dim) shape.",
    "<b>Cosine normalization:</b> the grader scores with cosine similarity, but the code "
    "used a raw dot product. Both the query and every candidate are now unit-normalized "
    "so dot product equals cosine similarity.",
    "<b>int32 IDs:</b> partition IDs were int64 (160 MB at 20M). Switched to int32 "
    "(80 MB) — 20M fits comfortably under int32's 2.1B ceiling.",
])

H2("6.3 IVF — searching less")
P("An IVF (inverted-file) index clusters all vectors into many groups using k-means. "
  "At query time, instead of scanning all 20M vectors, it scores the cluster centroids, "
  "picks the nearest <b>n_probe</b> clusters (32 of 4096), and only searches those — "
  "about 0.8% of the data. The index is stored in CSR (compressed-sparse-row) format:")
code(
"""ivf_index/
  centroids.npy          (n_clusters, dim) float32, unit-normalized
  partition_offsets.npy  (n_clusters+1,)   int64  CSR offsets
  partition_ids.npy      (n_total,)        int32  vector IDs, cluster-ordered
  partition_vectors.bin  (n_total, dim)    float32 normalized, cluster-ordered
""")

H2("6.4 The breakthrough — partition_vectors.bin")
P("The first IVF attempt was slower and used GIGABYTES of RAM. The cause: the ~156,000 "
  "candidate vectors were scattered randomly across a 5 GB memory-mapped file. Every "
  "random access made the OS prefetch ~256 KB \"just in case\", inflating resident "
  "memory to 2,333 MB. The fix was to pre-organize the data: <b>partition_vectors.bin</b> "
  "stores every vector contiguously, grouped by cluster. Now reading the 32 probe "
  "clusters is 32 sequential reads instead of 156,000 random page faults. Building it "
  "uses a single sequential pass over the source with scatter-writes (an inverse-mapping "
  "trick), which cut 20M index build time from ~6000 s to ~120 s.")

H2("6.5 Zero-allocation queries")
P("To drive per-query RAM to near zero, the index vectors are loaded fully into RAM "
  "once at init, and the candidate buffer, IDs buffer, and scores buffer are "
  "pre-allocated. Each query fills those existing buffers in place "
  "(np.dot(..., out=score_buf)) — nothing new is allocated, so the memory profiler "
  "measures ~0 MB of growth during a query.")

H2("6.6 Final results")
table([
    ["DB size", "Time", "Per-query RAM", "Score"],
    ["1M", "0.01 s", "0.08 MB", "0.0 (perfect)"],
    ["10M", "0.02 s", "0.00 MB", "0.0 (perfect)"],
    ["20M", "0.04 s", "0.00 MB", "0.0 (perfect)"],
], [3.5 * cm, 3.5 * cm, 4.5 * cm, 4.5 * cm])
P("Accuracy stayed perfect throughout — the speed and memory gains cost nothing in "
  "result quality, because real embeddings cluster well and probing 32 clusters keeps "
  "the true nearest neighbors in range.")

# ---- 7. Storage layer ----
br()
H1("7. Storage Layer")
H2("VecDBStore (core/storage/vecdb_store.py)")
P("A drop-in vector store that wraps VecDB. It accumulates vectors in an in-memory "
  "matrix and saves to .npy. For small collections it runs direct cosine search; after "
  "build_index() it loads the IVF index and switches to fast IVF search automatically. "
  "It supports soft-delete (mark a row deleted, exclude from search/count) and compact() "
  "(physically drop deleted rows and rebuild).")
H2("MetadataStore (core/storage/metadata_store.py)")
P("Stores one JSON record per memory: a UUID id, the text, a metadata dict "
  "(importance, retrieval_count, source info, etc.), and a UTC created_at timestamp. "
  "Soft-delete sets a deleted flag; compact() removes flagged records and returns the "
  "surviving indices so the vector store can stay in sync.")
P("Both stores read their paths from configuration (Settings), so deployments and tests "
  "can isolate storage with environment variables.")

# ---- 8. Sprint 0 ----
H1("8. Sprint 0 — Wiring the Engine In")
P("Before Sprint 0 there were two disconnected systems: the optimized vec_db.py, and a "
  "separate toy in-memory VectorStore that the service actually used. Sprint 0 replaced "
  "the toy with <b>VecDBStore</b>, so the fast engine became the real backend. It also "
  "fixed a bug where every /search request constructed a brand-new MemoryService (re-"
  "loading all data), and restored a missing hybrid-retrieval file that was crashing "
  "imports.")

# ---- 9. Sprint 1 ----
H1("9. Sprint 1 — Memory Intelligence")
P("Memory that only grows becomes noise. Sprint 1 added lifecycle management:")
bullets([
    "<b>Consolidation / merge:</b> detect near-duplicate memories by cosine similarity, "
    "and merge two into one re-encoded combined memory.",
    "<b>Pruning:</b> score each memory by recency (exponential decay) + importance + "
    "access frequency; soft-delete those below a threshold. Memories that have been "
    "accessed or flagged important are protected.",
    "<b>Summarization:</b> collapse a cluster of related memories into one. Sprint 1 is "
    "extractive (pick the most central memory + context); Sprint 3's LLM can make it "
    "abstractive.",
    "<b>Soft-delete + compact:</b> nothing is destroyed until compact() runs, so deletes "
    "are reversible up to that point.",
])
P("Supporting pieces: <b>decay.py</b> (half-life recency scoring, slower decay for "
  "important memories), <b>ranking.py</b> (weighted blend of similarity, recency, "
  "importance, frequency).")

# ---- 10. Sprint 2 ----
H1("10. Sprint 2 — Document & Codebase Ingestion")
P("Sprint 2 lets NeuralVault learn from real documents. All loaders share the same "
  "TextChunker (paragraph-aware splitting with a size budget) and return a uniform "
  "list of {text, metadata} chunks.")
table([
    ["Loader", "Handles", "Notes"],
    ["NotebookLoader", ".ipynb", "markdown + code cells"],
    ["PDFLoader", ".pdf", "page-by-page via pypdf"],
    ["DOCXLoader", ".docx", "paragraphs via python-docx"],
    ["DirectoryLoader", "folders", "recursive; skips .git/.venv/__pycache__/binaries; "
                                    "tags code vs text"],
], [3.6 * cm, 2.4 * cm, 10 * cm])

# ---- 11. Sprint 3 ----
br()
H1("11. Sprint 3 — LLM Integration")
P("Sprint 3 turns the vector database into an actual memory layer for an LLM. Three "
  "pieces:")
H2("ContextBuilder (core/llm/context_builder.py)")
P("Formats retrieved memories into a numbered, source-tagged block, bounded by a "
  "character budget. It builds the system prompt as two blocks: a frozen instruction "
  "block carrying a cache_control marker (so the instruction prefix is cached across "
  "requests) followed by the volatile memory context. This prefix-caching layout cuts "
  "cost on repeated calls.")
H2("ConversationMemory (core/llm/conversation.py)")
P("Holds multi-turn history as Messages-API dicts, trimmed to a max number of turns "
  "while always preserving a leading user message, so follow-up questions retain context.")
H2("LLMClient (core/llm/llm_client.py)")
P("Wraps the Anthropic SDK: Claude Opus 4.8, adaptive thinking, streaming assembled via "
  "get_final_message(). The anthropic import is lazy and the client is injectable, so "
  "the whole stack imports and tests fully offline. If no API key is configured it "
  "raises a clear LLMUnavailableError instead of crashing.")
H2("chat()")
P("The MemoryService.chat() method ties it together: retrieve top-k memories -> build "
  "the cached system prompt -> call the LLM -> record the turn in conversation memory -> "
  "return the answer plus <b>memory attribution</b> (each memory's index, UUID, text, "
  "and relevance score) for explainability.")

# ---- 12. Sprint 4 ----
H1("12. Sprint 4 — Production Hardening")
bullets([
    "<b>Config (utils/config.py):</b> a Settings dataclass sourced from environment "
    "variables with defaults; one place for all configuration.",
    "<b>Metrics (utils/metrics.py):</b> thread-safe per-operation request counts, error "
    "counts, and average latency, with a timer() context manager.",
    "<b>Middleware:</b> times every HTTP request, records metrics, adds an "
    "X-Process-Time header; a global exception handler returns a structured 500 instead "
    "of leaking a stack trace.",
    "<b>Health endpoints:</b> /health (liveness), /health/ready (readiness + whether the "
    "LLM key is present), /metrics (live snapshot).",
    "<b>Graceful degradation:</b> without an API key the service still runs (ingest, "
    "search, manage); only /memory/chat returns 503 with a helpful message.",
])

# ---- 13. CI/CD, Docker, Frontend ----
H1("13. CI/CD, Docker, and Frontend")
H2("Continuous Integration")
P("A GitHub Actions workflow (.github/workflows/test.yml) runs ruff lint and the full "
  "pytest suite on every push and pull request, with pip and HuggingFace model caching "
  "so the embedding model is not re-downloaded each run. Confirmed green: fresh clone -> "
  "install -> lint -> test -> pass.")
H2("Docker")
P("A slim, non-root Dockerfile with a HEALTHCHECK runs the app under uvicorn. "
  "docker-compose.yml bind-mounts ./data and ./logs for persistence and caches the "
  "embedding model in a named volume — `docker compose up -d` runs the whole stack.")
H2("Frontend")
P("A single-file demo UI (frontend/index.html, vanilla JS, no build step) is served by "
  "FastAPI at /ui on the same origin (so no CORS). Three tabs: <b>Chat</b> (answers + "
  "attribution bars), <b>Memory Explorer</b> (add / search / browse / delete / inspect), "
  "and <b>Dashboard</b> (health + metrics).")

# ---- 14. End-to-end ----
br()
H1("14. End-to-End: A Chat Request")
P("What happens when a user asks a question through /memory/chat:")
code(
"""1. POST /memory/chat {"query": "when is the deadline?", "top_k": 5}
2. Middleware starts a timer; route calls MemoryService.chat().
3. TextEncoder embeds the query  -> 384-dim unit vector.
4. VecDBStore.search() returns the top-5 memories
   (brute force, or IVF if an index was built); deleted rows skipped.
5. SemanticRetriever re-ranks by similarity + recency + importance + frequency,
   increments each memory's retrieval_count.
6. ContextBuilder builds a 2-block system prompt:
   [frozen instructions | cache_control]  +  [retrieved memories].
7. ConversationMemory appends the user turn (prior history included).
8. LLMClient streams Claude Opus 4.8 with adaptive thinking;
   get_final_message() assembles the answer.
9. The assistant turn is recorded; response returns:
   { answer, memories_used:[{index,id,text,score}], usage }
10. Middleware records latency into Metrics; /metrics now reflects the call.
""")
P("Without an API key, steps 1–5 still run; step 8 raises LLMUnavailableError and the "
  "route returns HTTP 503.")

# ---- 15. API reference ----
H1("15. API Reference")
table([
    ["Method & path", "Purpose"],
    ["POST /memory/add", "Store one memory"],
    ["POST /memory/add-batch", "Store many memories"],
    ["POST /memory/search", "Semantic top-k retrieval"],
    ["GET  /memory/list", "Browse memories (paginated)"],
    ["GET  /memory/count", "Count non-deleted memories"],
    ["DELETE /memory/{index}", "Soft-delete a memory"],
    ["POST /memory/compact", "Physically remove deleted memories"],
    ["POST /memory/build-index", "Build the IVF index for scale"],
    ["POST /memory/chat", "LLM answer grounded in memories"],
    ["POST /memory/chat/reset", "Clear conversation history"],
    ["POST /memory/prune", "Remove low-value memories"],
    ["POST /memory/summarize", "Summarize a set of memories"],
    ["POST /memory/consolidation/candidates", "Find duplicate pairs"],
    ["POST /memory/consolidation/merge", "Merge two memories"],
    ["POST /memory/ingest/{notebook,pdf,docx,directory}", "Ingest documents"],
    ["GET  /health  ·  /health/ready  ·  /metrics", "Ops / monitoring"],
    ["GET  /ui", "Demo web UI"],
], [8.2 * cm, 7.8 * cm])

# ---- 16. Config reference ----
H1("16. Configuration Reference")
P("All settings are environment variables (see .env.example). Only ANTHROPIC_API_KEY is "
  "required, and only for chat.")
table([
    ["Variable", "Default", "Purpose"],
    ["ANTHROPIC_API_KEY", "(none)", "Enables /memory/chat"],
    ["NEURALVAULT_LLM_MODEL", "claude-opus-4-8", "Generation model"],
    ["NEURALVAULT_LLM_MAX_TOKENS", "4096", "Max answer length"],
    ["NEURALVAULT_DEFAULT_TOP_K", "5", "Default retrieval depth"],
    ["NEURALVAULT_MAX_CONTEXT_CHARS", "8000", "Context budget"],
    ["NEURALVAULT_DB_PATH", "data/embeddings/vectors.npy", "Vector store path"],
    ["NEURALVAULT_INDEX_PATH", "data/indexes/memory_ivf", "IVF index path"],
    ["NEURALVAULT_METADATA_PATH", "data/processed/metadata.json", "Metadata path"],
], [6 * cm, 5.5 * cm, 4.5 * cm])

# ---- 17. Testing ----
br()
H1("17. Testing")
P("The suite has <b>100 tests</b>, all passing and fully offline — the LLM is mocked "
  "end-to-end with a fake Anthropic client, so no API key or network is needed. Tests "
  "cover the vector store, retrieval, every sprint's features, the API endpoints, and "
  "the frontend-supporting routes.")
P("<b>Test isolation</b> is enforced by tests/conftest.py, which points all storage "
  "paths at a throwaway temp directory at import time — before any test or the app "
  "singleton loads — so no test can ever touch the real data directory. This was added "
  "after a fixture leak corrupted the local store and broke app startup; it is now "
  "structurally impossible.")
code(
"""pytest                                       # 100 tests, offline
ruff check core api main.py vec_db.py tests  # lint
""")

# ---- 18. Running ----
H1("18. Running NeuralVault")
H2("Local")
code(
"""pip install -r requirements-dev.txt
uvicorn main:app --reload          # http://localhost:8000
# API docs:  /docs        Demo UI:  /ui
""")
H2("Docker")
code(
"""cp .env.example .env               # add ANTHROPIC_API_KEY (optional)
docker compose up -d               # persists ./data and ./logs
""")

# ---- 19. Lessons ----
H1("19. Engineering Decisions & Lessons")
bullets([
    "<b>Correctness before speed.</b> The IVF work was worthless until the cosine-"
    "normalization and .dat-shape bugs were fixed. A fast wrong answer is still wrong.",
    "<b>Sequential beats random.</b> The single biggest win was data layout — storing "
    "vectors cluster-contiguously turned 156k random page faults into 32 sequential "
    "reads, collapsing RAM from 2.3 GB to ~0.",
    "<b>Measure what the grader measures.</b> Per-query RAM looked fine until profiled "
    "correctly; the zero-allocation buffers were designed against the actual metric.",
    "<b>CI is a safety net that catches what narrow runs miss.</b> Running the full "
    "suite (not just sprint files) surfaced a clobbered hybrid-retrieval file and a "
    "duplicate method — both real regressions.",
    "<b>Config gaps hide data bugs.</b> Stores ignoring their configured paths let test "
    "fixtures leak into real data and corrupt it. Making storage config-driven fixed the "
    "bug and closed the gap at once.",
    "<b>Secrets never reach the repo.</b> GitHub push protection blocked a commit that "
    "swept in a notebook containing a personal access token; the notebook was untracked "
    "and history confirmed clean.",
])

# ---- 20. Glossary ----
H1("20. Glossary")
table([
    ["Term", "Meaning"],
    ["Embedding", "A vector of numbers representing the meaning of text."],
    ["Cosine similarity", "Similarity of two vectors by the angle between them."],
    ["IVF", "Inverted-file index: cluster vectors, search only nearby clusters."],
    ["n_probe", "How many clusters to search per query (recall vs speed)."],
    ["CSR", "Compressed-sparse-row layout for the cluster -> IDs mapping."],
    ["Soft-delete", "Mark a record deleted without removing it until compaction."],
    ["Prompt caching", "Reusing a cached prompt prefix to cut cost/latency."],
    ["Adaptive thinking", "Claude deciding how much to reason per request."],
    ["Top-k", "The k most similar results returned for a query."],
], [4 * cm, 12 * cm])

gap(10)
P("<i>NeuralVault — built in five phases on a hand-optimized vector database, "
  "from 2.58 s brute force to 0.04 s grounded recall.</i>")


# --------------------------------------------------------------------------- #
def main():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="NeuralVault — Technical Documentation",
        author="NeuralVault",
    )
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
