# AutoMark

**Student Assignment Auto-Grader Multi-Agent System**

A local, privacy-preserving auto-grader built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com). It evaluates student submissions (`.txt` or `.pdf`) against a JSON rubric using a pipeline of specialised agents, producing a structured Markdown feedback report and a marking sheet — entirely on your own hardware with no external API calls.

A FastAPI REST wrapper exposes the pipeline as an HTTP service, making it easy to integrate with other tools or front-ends.

---

## Table of Contents

- [Architecture](#architecture)
- [Agents](#agents)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [REST API](#rest-api)
  - [Docker Stack](#docker-stack)
- [Make Targets](#make-targets)
- [Testing](#testing)
- [Data Formats](#data-formats)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

---

## Architecture

The pipeline is a directed LangGraph `StateGraph`. The entry point routes to the appropriate ingestion agent based on the submission file type, then proceeds through analysis, historical comparison, and report generation:

```
                     ┌── ingestion (txt) ──┐
_detect ─────────────┤                     ├──► analysis ──► historical ──► report ──► END
                     └── pdf_ingestion ────┘       │                           ▲
                                                   └── (ingestion failed) ─────┘
```

| Step | Agent | Responsibility |
|---|---|---|
| 1 | **_detect** | Routes to the correct ingestion agent based on file extension |
| 2a | **Ingestion** | Validates and reads `.txt` submissions; extracts `student_id` from text |
| 2b | **PDF Ingestion** | Converts `.pdf` to Markdown via `pymupdf4llm`; uses the LLM to extract `student_id` and `student_name` |
| 3 | **Analysis** | Scores each rubric criterion with the LLM; totals computed deterministically |
| 4 | **Historical** | Persists results to SQLite; retrieves past reports; generates progression insights |
| 5 | **Report** | Generates a Markdown feedback report and a marking sheet via LLM |

If ingestion fails (e.g. missing files), the pipeline short-circuits directly to `report`, which produces a minimal fallback report without scores.

---

## Agents

### Ingestion Agent (`agents/ingestion.py`)
Validates that both input file paths are non-empty, the files exist, are non-empty, and have the correct extensions. Reads the plain-text submission and parses the rubric JSON. Extracts `student_id` from the submission text using a regex pattern (`Student ID: <value>`). Sets `ingestion_status` to `"success"` or `"failed"`.

### PDF Ingestion Agent (`agents/pdf_ingestion.py`)
Validates the `.pdf` submission path, converts the PDF to Markdown using `pymupdf4llm`, and always passes that full Markdown to downstream scoring/reporting. The light LLM is used only for extracting student metadata (`student_id`, `student_name`) from a compact metadata-focused prompt. Sets `ingestion_status` to `"success"` or `"failed"`.

### Analysis Agent (`agents/analysis.py`)
Calls `phi4-mini` via LangChain/Ollama to score each rubric criterion. LLM output is structured using a Pydantic schema (`RubricScores`). Scores are clamped to `[0, max_score]` and the total is computed deterministically by `calculate_total_score` — the LLM is never trusted for arithmetic.

### Historical Agent (`agents/historical.py`)
Saves the current grading result to a SQLite database, retrieves all previous reports for the student, and uses the LLM to generate concise progression insights when past reports exist. Also writes a separate performance analysis report to disk.

### Report Agent (`agents/report.py`)
Calls `phi4-mini` to generate a well-formatted Markdown feedback report and a marking sheet. Falls back to a template-based report if the LLM is unavailable.

---

## Tools

| Module | Function | Description |
|---|---|---|
| `tools/file_ops.py` | `read_text_file` | Reads a UTF-8 text file; wraps `OSError` as `RuntimeError` |
| `tools/file_ops.py` | `read_json_file` | Reads and parses a JSON file; wraps `OSError`/`JSONDecodeError` |
| `tools/file_ops.py` | `validate_submission_files` | Checks existence, size, and file extensions |
| `tools/file_writer.py` | `write_feedback_report` | Writes feedback report to disk, creating parent directories as needed |
| `tools/file_writer.py` | `write_analysis_report` | Writes the performance analysis report to disk |
| `tools/score_calculator.py` | `calculate_total_score` | Sums criterion scores, computes percentage, assigns a letter grade |
| `tools/logger.py` | `log_agent_action`, `log_model_call`, `timed_model_call` | Emits structured logs via `structlog` (JSON file + console output) |
| `tools/db_manager.py` | `init_db` | Initialises the SQLite student results database |
| `tools/db_manager.py` | `save_report` | Persists a grading result for a student |
| `tools/db_manager.py` | `get_past_reports` | Retrieves all previous grading results for a student |
| `tools/pdf_processor.py` | `convert_pdf_to_markdown` | Converts a PDF file to Markdown text using `pymupdf4llm` |

**Grade thresholds:**

| Grade | Percentage |
|---|---|
| A | ≥ 90% |
| B | ≥ 75% |
| C | ≥ 60% |
| D | ≥ 50% |
| F | < 50% |

---

## Project Structure

```
.
├── data/
│   ├── rubric.json          # Rubric definition (criteria + max scores)
│   ├── submission.txt       # Sample plain-text student submission
│   └── students.db          # SQLite database (created by `make init-db`)
├── src/mas/
│   ├── agents/
│   │   ├── ingestion.py     # Plain-text ingestion + student ID extraction
│   │   ├── pdf_ingestion.py # PDF → Markdown ingestion + LLM detail extraction
│   │   ├── analysis.py      # LLM-powered rubric scoring
│   │   ├── historical.py    # Persist results + progression insights
│   │   └── report.py        # Markdown feedback report + marking sheet
│   ├── tools/
│   │   ├── file_ops.py      # File reading and validation helpers
│   │   ├── file_writer.py   # Report and analysis file writers
│   │   ├── score_calculator.py
│   │   ├── logger.py
│   │   ├── db_manager.py    # SQLite persistence helpers
│   │   └── pdf_processor.py # pymupdf4llm PDF-to-Markdown converter
│   ├── api.py               # FastAPI REST wrapper
│   ├── config.py            # Environment-based settings (python-dotenv)
│   ├── graph.py             # LangGraph pipeline definition
│   ├── llm.py               # Ollama LLM factory functions
│   └── state.py             # Shared AgentState TypedDict
├── tests/
│   ├── test_tools.py        # Unit tests for all tool modules (no LLM)
│   ├── test_ingestion.py    # Ingestion agent tests
│   ├── test_pdf_ingestion.py# PDF ingestion agent tests (mocked LLM)
│   ├── test_analysis.py     # Analysis agent tests (mocked LLM)
│   ├── test_historical.py   # Historical agent tests (mocked LLM + DB)
│   ├── test_report.py       # Report agent tests (mocked LLM)
│   ├── test_config.py       # Settings / config tests
│   └── test_llm_judge.py    # LLM-as-a-Judge tests via phi4-mini + requests
├── output/                  # Generated reports (created on first run)
├── Dockerfile.api           # Docker image for the FastAPI service
├── docker-compose.yml       # Full stack: Ollama + AutoMark API
├── .env.example             # Example environment variable file
├── Makefile
└── pyproject.toml
```

---

## Prerequisites

- **Python 3.13+** and [uv](https://docs.astral.sh/uv/) (package manager)
- **Docker** with Docker Compose (for the Ollama service and/or full stack)
- At least **4 GB of free RAM** for the `phi4-mini:3.8b-q4_K_M` model
- A GPU with NVIDIA drivers is optional but recommended for faster inference

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/nmdra/AutoMark.git
cd AutoMark
```

**2. Install Python dependencies**

```bash
# Runtime + dev dependencies (includes pytest and requests)
uv pip install -e ".[dev]"
```

**3. Configure environment variables** *(optional)*

```bash
cp .env.example .env
# Edit .env to override defaults (model name, paths, Ollama URL, etc.)
```

**4. Initialise the student database**

```bash
make init-db
```

**5. Start the Ollama service**

```bash
# CPU (default)
make start

# GPU (NVIDIA)
make start-gpu
```

**6. Pull the model**

```bash
make pull-model
```

This pulls `phi4-mini:3.8b-q4_K_M` into the Ollama container. The download is ~2.5 GB and only needs to be done once (data is persisted in the `~/.ollama` volume mount).

---

## Configuration

Settings are loaded from environment variables (or a `.env` file in the project root). All values have sensible defaults.

| Variable | Default | Description |
|---|---|---|
| `AUTOMARK_MODEL_NAME` | `phi4-mini:3.8b-q4_K_M` | Ollama model identifier |
| `AUTOMARK_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama HTTP API base URL |
| `AUTOMARK_DB_PATH` | `data/students.db` | SQLite database path |
| `AUTOMARK_LOG_FILE` | `agent_trace.log` | JSON agent trace log path |
| `AUTOMARK_OUTPUT_PATH` | `output/feedback_report.md` | Default feedback report path |
| `AUTOMARK_MARKING_SHEET_PATH` | `output/marking_sheet.md` | Default marking sheet path |
| `AUTOMARK_ANALYSIS_REPORT_PATH` | `output/analysis_report.md` | Default analysis report path |
| `AUTOMARK_DATA_BASE_DIR` | `<project_root>/data` | Base directory for submission/rubric files (API path-traversal guard) |
| `AUTOMARK_JOB_WORKER_CONCURRENCY` | `2` | Worker threads for async batch jobs |
| `AUTOMARK_JOB_QUEUE_MAX_SIZE` | `100` | Max queued batch jobs in memory |
| `AUTOMARK_JOB_MAX_RETRIES` | `1` | Default retries per batch item |
| `AUTOMARK_BATCH_MAX_ITEMS` | `100` | Max items accepted per batch request |
| `AUTOMARK_JOB_RETENTION_DAYS` | `30` | Suggested retention period for completed jobs |
| `AUTOMARK_EXPORT_MAX_BYTES` | `10485760` | Max allowed CSV/JSON/PDF export size (bytes) |

---

## Usage

### REST API

Start the development API server (connects to Ollama at `localhost:11434`):

```bash
make api
```

The interactive API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/grade` | Run the grading pipeline |
| `POST` | `/grade/batch` | Submit asynchronous batch grading job |
| `GET` | `/jobs` | List async jobs (supports status, limit, offset) |
| `GET` | `/jobs/{job_id}` | Get full job status with per-item results |
| `POST` | `/jobs/{job_id}/cancel` | Cancel queued/running job |
| `POST` | `/jobs/{job_id}/exports/{format}` | Generate CSV/JSON/PDF export for completed job |
| `GET` | `/jobs/{job_id}/exports/{format}` | Download previously generated export |
| `GET` | `/sessions/{session_id}/logs` | Retrieve trace log entries for a session |

**Example grading request:**

```bash
curl -X POST http://localhost:8000/grade \
  -H "Content-Type: application/json" \
  -d '{"submission_path": "submission.txt", "rubric_path": "rubric.json"}'
```

Both paths are resolved relative to `AUTOMARK_DATA_BASE_DIR` (default: `data/`). Path traversal is blocked.

**Example batch request:**

```bash
curl -X POST http://localhost:8000/grade/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"submission_path": "submission.txt", "rubric_path": "rubric.json", "correlation_id": "s1"},
      {"submission_path": "submission2.txt", "rubric_path": "rubric.json", "correlation_id": "s2"}
    ],
    "max_retries": 1
  }'
```

Then poll `GET /jobs/{job_id}` for status/progress and use
`POST /jobs/{job_id}/exports/{format}` (`csv`, `json`, or `pdf`) to build downloadable artifacts.

**Example response (abbreviated):**

```json
{
  "session_id": "abc123",
  "student_id": "IT21000001",
  "student_name": "",
  "total_score": 17.0,
  "percentage": 85.0,
  "grade": "B",
  "summary": "A well-structured submission...",
  "criteria": [...],
  "feedback_report": "## Feedback\n...",
  "output_filepath": "output/20240101_120000_..._feedback_report.md",
  "marking_sheet_path": "output/20240101_120000_..._marking_sheet.md"
}
```

### Docker Stack

To run the full stack (Ollama + AutoMark API) with Docker Compose:

```bash
make docker-up
```

This builds the API image and starts both the Ollama and AutoMark API containers. The API is available at `http://localhost:8000`. Stop with:

```bash
make docker-down
```

---

## Make Targets

| Target | Description |
|---|---|
| `make start` | Start Ollama with the CPU Docker profile |
| `make start-gpu` | Start Ollama with the NVIDIA GPU Docker profile |
| `make stop` | Stop all Ollama containers |
| `make pull-model` | Pull `phi4-mini:3.8b-q4_K_M` into the Ollama container |
| `make init-db` | Initialise the SQLite student database at `data/students.db` |
| `make api` | Start the FastAPI development server on port 8000 |
| `make docker-up` | Build and start the full Docker stack (Ollama + API) |
| `make docker-down` | Stop the full Docker stack |
| `make test` | Run the full pytest test suite |
| `make logs` | Tail the Ollama container logs |
| `make clean` | Remove containers, volumes, `__pycache__`, and generated output |

---

## Testing

```bash
make test
# or directly:
uv run pytest tests/ -v
```

The test suite is split into three layers:

### Tool unit tests (`test_tools.py`)
Fast, deterministic tests covering all tool modules — no LLM or network required. They test success paths, error handling, edge cases, file I/O, grade boundary thresholds, log format, and timestamp validity.

### Agent integration tests
Each agent has its own test file. LLM calls are mocked with `unittest.mock`, so these tests run instantly without Ollama:

| File | Agent under test |
|---|---|
| `test_ingestion.py` | Ingestion — file validation, student ID extraction, error handling |
| `test_pdf_ingestion.py` | PDF Ingestion — PDF conversion, LLM extraction, fallback behaviour |
| `test_analysis.py` | Analysis — scoring, score clamping, LLM fallback, grade logic |
| `test_historical.py` | Historical — DB persistence, past report retrieval, insights generation |
| `test_report.py` | Report — file writing, LLM fallback, overwrite behaviour |
| `test_config.py` | Config — environment variable loading, defaults |

### LLM-as-a-Judge (`test_llm_judge.py`)
Uses the `requests` library to call `phi4-mini` directly via the Ollama REST API (`/api/generate`) and ask whether a set of assigned scores is fair. The model is prompted to respond with **only** `YES` or `NO`.

Three tests are included:
- Fair, high scores on a strong submission → expects `YES`
- Zero scores on the same strong submission → expects `NO`
- Response format assertion (must be exactly `YES` or `NO`)

These tests are **automatically skipped** when Ollama is not running or the model has not been pulled.

---

## Data Formats

### Submission

Supported formats:
- **Plain text** (`.txt`) — UTF-8 encoded. Optionally include `Student ID: <value>` on any line for automatic ID extraction.
- **PDF** (`.pdf`) — The PDF is converted to Markdown automatically. The LLM attempts to extract `student_id` and `student_name` from the cover page.

### Rubric (`data/rubric.json`)

```json
{
  "module": "CTSE – IT4080",
  "assignment": "Cloud Technology Fundamentals",
  "total_marks": 20,
  "criteria": [
    {
      "id": "C1",
      "name": "Definition of Containerisation",
      "description": "Accurately defines containerisation and distinguishes it from virtual machines.",
      "common_mistakes": ["missing_answer", "out_of_context"],
      "max_score": 5
    }
  ]
}
```

Required fields: `total_marks` (integer), `criteria` (array). Each criterion requires `id`, `name`, `description`, and `max_score`.  
Optional field: `common_mistakes` (array) to document expected mistakes such as `missing_answer` and `out_of_context`.

---

## Output

After a successful run, the following files are written to the `output/` directory. When called via the REST API, filenames include a timestamp, student name, and student ID to prevent collisions:

| File | Description |
|---|---|
| `*_feedback_report.md` | Overall summary, per-criterion scores and justifications, improvement suggestions |
| `*_marking_sheet.md` | Compact marking sheet including per-criterion common-mistake tags and a common-mistakes summary |
| `*_analysis_report.md` | Historical performance analysis with progression insights |

An append-only structured JSON trace log (via `structlog`) is written to `agent_trace.log` in the project root on every run.

---

## Troubleshooting

**Ollama is not reachable**
Ensure the container is running (`docker ps`) and the port is bound: `curl http://localhost:11434/api/tags`

**Model not found**
Run `make pull-model`. The first pull requires ~2.5 GB of disk space.

**LLM returns malformed JSON (Analysis Agent)**
The agent catches this and falls back to zero scores for all criteria with an `error` field in the final state. Check `agent_trace.log` for details.

**GPU not detected**
Make sure the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) is installed and use `make start-gpu`.

**Database not initialised**
Run `make init-db` before the first grading run to create `data/students.db`.
