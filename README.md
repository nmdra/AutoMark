# ctse-mas

**Student Assignment Auto-Grader Multi-Agent System**

A local, privacy-preserving auto-grader built with [LangGraph](https://github.com/langchain-ai/langgraph) and [Ollama](https://ollama.com). It evaluates student submissions against a JSON rubric using a pipeline of specialised agents, producing a structured Markdown feedback report — entirely on your own hardware with no external API calls.

---

## Table of Contents

- [Architecture](#architecture)
- [Agents](#agents)
- [Tools](#tools)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Make Targets](#make-targets)
- [Testing](#testing)
- [Data Formats](#data-formats)
- [Output](#output)
- [Troubleshooting](#troubleshooting)

---

## Architecture

The pipeline is a directed LangGraph `StateGraph` with one conditional edge:

```
coordinator ──► research ──► analysis ──► report ──► END
                    │                       ▲
                    └── (research failed) ──┘
```

| Step | Agent | Responsibility |
|---|---|---|
| 1 | **Coordinator** | Validates file paths and extensions; assigns a session ID |
| 2 | **Research** | Reads submission `.txt` and rubric `.json` into shared state |
| 3 | **Analysis** | Scores each rubric criterion with the LLM; totals computed deterministically |
| 4 | **Report** | Generates a Markdown feedback report via LLM; writes it to disk |

If `research` fails (e.g. missing files), the pipeline short-circuits directly to `report`, which produces a minimal fallback report without scores.

---

## Agents

### Coordinator Agent (`agents/coordinator.py`)
Validates that both input file paths are non-empty, the files exist, are non-empty, and have the correct extensions (`.txt` for submission, `.json` for rubric). Generates a UUID session ID if one is not provided.

### Research Agent (`agents/research.py`)
Reads the plain-text submission and parses the JSON rubric into `AgentState`. Never scores or evaluates content.

### Analysis Agent (`agents/analysis.py`)
Calls `phi4-mini` via LangChain/Ollama to score each rubric criterion. LLM output is structured using a Pydantic schema (`RubricScores`). Scores are clamped to `[0, max_score]` and the total is computed deterministically by `calculate_total_score` — the LLM is never trusted for arithmetic.

### Report Agent (`agents/report.py`)
Calls `phi4-mini` to generate a well-formatted Markdown feedback report. Falls back to a template-based report if the LLM is unavailable.

---

## Tools

| Module | Function | Description |
|---|---|---|
| `tools/file_reader.py` | `read_text_file` | Reads a UTF-8 text file; wraps `OSError` as `RuntimeError` |
| `tools/file_reader.py` | `read_json_file` | Reads and parses a JSON file; wraps `OSError`/`JSONDecodeError` |
| `tools/file_validator.py` | `validate_submission_files` | Checks existence, size, and file extensions |
| `tools/file_writer.py` | `write_feedback_report` | Writes report to disk, creating parent directories as needed |
| `tools/score_calculator.py` | `calculate_total_score` | Sums criterion scores, computes percentage, assigns a letter grade |
| `tools/logger.py` | `log_agent_action` | Appends a JSON trace entry to `agent_trace.log` |

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
│   └── submission.txt       # Sample student submission
├── src/ctse_mas/
│   ├── agents/
│   │   ├── coordinator.py
│   │   ├── research.py
│   │   ├── analysis.py
│   │   └── report.py
│   ├── tools/
│   │   ├── file_reader.py
│   │   ├── file_validator.py
│   │   ├── file_writer.py
│   │   ├── score_calculator.py
│   │   └── logger.py
│   ├── graph.py             # LangGraph pipeline + CLI entry point
│   ├── llm.py               # Ollama LLM factory functions
│   └── state.py             # Shared AgentState TypedDict
├── tests/
│   ├── test_tools.py        # Unit tests for all tool modules (no LLM)
│   ├── test_coordinator.py  # Coordinator agent integration tests
│   ├── test_research.py     # Research agent integration tests
│   ├── test_analysis.py     # Analysis agent integration tests (mocked LLM)
│   ├── test_report.py       # Report agent integration tests (mocked LLM)
│   └── test_llm_judge.py    # LLM-as-a-Judge tests via phi4-mini + requests
├── output/
│   └── feedback_report.md   # Generated after running the pipeline
├── docker-compose.yml       # Ollama service (CPU and GPU profiles)
├── Makefile
└── pyproject.toml
```

---

## Prerequisites

- **Python 3.11+** and [uv](https://docs.astral.sh/uv/) (package manager)
- **Docker** with Docker Compose (for the Ollama service)
- At least **4 GB of free RAM** for the `phi4-mini:3.8b-q4_0` model
- A GPU with NVIDIA drivers is optional but recommended for faster inference

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/nmdra/AI-Agent.git
cd AI-Agent
```

**2. Install Python dependencies**

```bash
# Runtime + dev dependencies (includes pytest and requests)
uv pip install -e ".[dev]"
```

**3. Start the Ollama service**

```bash
# CPU (default)
make start

# GPU (NVIDIA)
make start-gpu
```

**4. Pull the model**

```bash
make pull-model
```

This pulls `phi4-mini:3.8b-q4_0` into the Ollama container. The download is ~2.5 GB and only needs to be done once (data is persisted in the `ollama_data` Docker volume).

---

## Usage

Run the auto-grader against the sample submission and rubric in `data/`:

```bash
make run
```

Example output:

```
=== Auto-Grader Complete ===
Grade  : B
Score  : 17
Report : /path/to/output/feedback_report.md
```

The full Markdown feedback report is written to `output/feedback_report.md`.

To grade a custom submission, edit `data/submission.txt` and `data/rubric.json`, or modify the paths in `src/ctse_mas/graph.py`.

---

## Make Targets

| Target | Description |
|---|---|
| `make start` | Start Ollama with the CPU Docker profile |
| `make start-gpu` | Start Ollama with the NVIDIA GPU Docker profile |
| `make stop` | Stop all Ollama containers |
| `make pull-model` | Pull `phi4-mini:3.8b-q4_0` into the Ollama container |
| `make run` | Run the full auto-grader pipeline |
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
48 fast, deterministic tests covering all tool modules — no LLM or network required. They test success paths, error handling, edge cases, file I/O, grade boundary thresholds, log format, and timestamp validity.

### Agent integration tests
Each agent has its own test file. LLM calls are mocked with `unittest.mock`, so these tests run instantly without Ollama:

| File | Agent under test |
|---|---|
| `test_coordinator.py` | Coordinator — validation, session ID, error handling |
| `test_research.py` | Research — file reading, error propagation, log entries |
| `test_analysis.py` | Analysis — scoring, score clamping, LLM fallback, grade logic |
| `test_report.py` | Report — file writing, LLM fallback, overwrite behaviour |

### LLM-as-a-Judge (`test_llm_judge.py`)
Uses the `requests` library to call `phi4-mini` directly via the Ollama REST API (`/api/generate`) and ask whether a set of assigned scores is fair. The model is prompted to respond with **only** `YES` or `NO`.

Three tests are included:
- Fair, high scores on a strong submission → expects `YES`
- Zero scores on the same strong submission → expects `NO`
- Response format assertion (must be exactly `YES` or `NO`)

These tests are **automatically skipped** when Ollama is not running or the model has not been pulled.

---

## Data Formats

### Submission (`data/submission.txt`)
Plain UTF-8 text. No special formatting required.

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
      "max_score": 5
    }
  ]
}
```

Required fields: `total_marks` (integer), `criteria` (array). Each criterion requires `id`, `name`, `description`, and `max_score`.

---

## Output

After a successful run, `output/feedback_report.md` contains:

- **Overall summary** — 2–3 sentence overview of the submission
- **Per-criterion feedback** — score, max score, and justification for each rubric criterion
- **Improvement suggestions** — constructive, actionable advice

An append-only JSON trace log is written to `agent_trace.log` in the project root on every run.

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

