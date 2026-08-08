# Agentic Food Data Linkages Constructor — POC

LLM-assisted probabilistic record linkage: connects USDA FoodData Central (FDC) records
to Open Food Facts (OFF), using `splink` (DuckDB backend) for the probabilistic matching
and a small LLM to propose/iterate blocking rules and matching attributes — mirroring
the manual SME workflow this is meant to assist. See `PLAN.md` for the full design.

## Setup

```bash
uv sync
cp .env.example .env   # adjust if needed; sane CPU defaults are baked in
```

Requires `data/food.parquet` (Open Food Facts) to already be present.

## Pipeline

```bash
uv run scripts/01_download_fdc.py          # scrape+download FNDDS/SR Legacy/Foundation/Branded
uv run scripts/02_convert_parquet.py       # CSV -> Parquet, + flatten OFF text fields
uv run scripts/03_build_fdc_db.py          # data/fdc.duckdb: unified per-dataset views
uv run scripts/04_build_calibration.py     # Branded<->OFF gold pairs, train/holdout split
uv run scripts/05_run_blocking_agent.py --block yogurt   # (or beans)
uv run scripts/06_run_matching_agent.py --block yogurt
uv run scripts/07_run_splink_and_evaluate.py --block yogurt
```

Each agent-loop script logs every round to `data/artifacts/` for SME review.

## LLM backend

Every agent-loop script calls `get_llm_client()`, selected by `LLM_DEVICE`:

- `LLM_DEVICE=cpu` (default): launches a local `vllm serve` subprocess. **`vllm` is not
  a project dependency** — its CPU build isn't a normal PyPI wheel (the default `vllm`
  wheel bundles CUDA and expects an NVIDIA GPU) and must be installed separately
  following vLLM's CPU-backend instructions. An 8B model's CPU inference is also slow
  and memory-heavy; on a resource-constrained box, prefer `LLM_DEVICE=mock` for
  development and only switch to a real CPU/GPU backend when you actually need live LLM
  reasoning.
- `LLM_DEVICE=cuda` / `rocm`: same server, GPU launch flags — a one-variable switch,
  see `src/agentic_matching/config.py`.
- `LLM_DEVICE=mock`: offline, no server required. `llm/mock.py` implements the same
  `ChatClient` interface with deterministic keyword-mining heuristics, enough to
  exercise/test/demo the full pipeline end-to-end without any LLM installed. This is
  what `scripts/05-07` were run with in this repo's checked-in `data/artifacts/`.
- Set `LLM_BASE_URL` instead to point at an already-running OpenAI-compatible server
  (e.g. on a remote GPU box) rather than launching one locally.

## Known constraints on this dev box

This box has no usable GPU (4GB iGPU VRAM, not viable for an 8B model) and 27GB RAM.
The pipeline pre-filters to per-block subsets (`data/blocks/<block>_{fndds,off}.parquet`,
written once by `scripts/05_run_blocking_agent.py`) before splink ever runs, so splink
never touches the full ~4.66M-row OFF table directly — but the blocks are still
asymmetric (hundreds to low thousands of FNDDS records vs. tens of thousands of OFF
records), and a few things below turned out to matter at that scale. All are fixed, but
worth knowing if this pipeline is extended (larger blocks, a real LLM instead of the
mock, more attributes, etc.):

- **EM blocking on a skewed attribute alone.** `linking/splink_model.py`'s EM passes
  deliberately combine any attribute column with the search-text prefix condition
  (`block_on(col, "substr(l.search_text, 1, 4)")`) rather than blocking on the attribute
  alone — blocking on a skewed boolean attribute by itself (e.g. `is_greek`, False for
  the vast majority of records) pairs up tens of millions of records and was what
  originally froze this machine.
- **Uncapped exhaustive holdout scoring.** `linking/evaluate.py`'s holdout scoring uses
  an exhaustive (`1=1`) blocking rule, safe only because the holdout sample size is
  capped (`max_holdout_positives`, default 500) — Branded Foods republishes many rows
  per GTIN, so an uncapped category holdout can be tens of thousands of rows, and an
  exhaustive cross join at that scale is the other thing that froze this machine.
- **Nondeterministic keyword-mining samples.** `blocking/agent_loop.py::_sample_texts`
  now `ORDER BY fdc_id`/`code` explicitly. Without it, `LIMIT` alone left row order (and
  therefore which records get sampled for keyword mining, mock or real LLM) up to
  DuckDB's query plan — observed to swing a block's OFF-side size by 2-3x across runs
  of identical code.
- **Overly-broad mined keywords.** `llm/mock.py`'s keyword mining rejects any candidate
  keyword whose background-sampled catalog-wide frequency implies it would match more
  than ~15,000 records (`_background_doc_freq` / `_top_tokens`'s `max_estimated_matches`)
  — a plain stopword list doesn't generalize (new generic words like `protein`, `rice`,
  `black`, `green` kept slipping through and each independently matched 20K-65K OFF
  records), and a plain *relative* frequency threshold is too loose at OFF's ~4.66M-row
  scale (even a "rare" token implies a large absolute match count). This only affects
  the mock; a real LLM should reason about keyword breadth on its own, but if a proposed
  rule still produces an outsized block, check `data/blocks/<block>_off.parquet`'s row
  count before running the linking stage.

If `scripts/07_...` still runs long or memory climbs unbounded, stop it and check
`data/blocks/<block>_off.parquet` row counts — a much larger OFF-side block than the
ones checked in here may need tighter blocking before EM.

## Testing

```bash
uv run pytest
```
