# Agentic Food Data Linkages Constructor — POC

> This is the original planning document, written before implementation began, and is
> kept as a historical record of initial design intent — it isn't updated to track
> subsequent implementation decisions (e.g. the LLM backend below now runs on Ollama
> only; see README.md for the current, up-to-date architecture and setup).

## Context

We're building a POC of an "LLM-assisted probabilistic record linkage" pipeline that connects
USDA FoodData Central (FDC) records to an external product database (Open Food Facts, standing
in for Circana), using `splink` (DuckDB backend) for the actual probabilistic matching and a
small locally-hosted 8B LLM (via vLLM) to propose/iterate blocking and matching-attribute logic —
mirroring the manual process SMEs currently use for the Purchase to Plate Suite.

Scope for this POC, confirmed with the user:
- Download FNDDS, SR Legacy, Foundation Foods, and Branded Foods CSVs from FDC and convert to
  Parquet.
- Link everything linkable *within* USDA using the shared `fdc_id` key (food.csv joins to each
  dataset-specific table, nutrients, portions, attributes, categories).
- External data: `data/food.parquet` (Open Food Facts, ~4.66M rows, already downloaded).
- **Calibration/gold set**: Branded Foods (`gtin_upc`) ↔ OFF (`code`) matched by normalized
  UPC/GTIN. FNDDS has no UPC field, so it cannot supply gold matches directly. Per the user's
  confirmation, Branded↔OFF gold pairs are the calibration/held-out evaluation signal, and the
  blocking/matching methodology calibrated on them is then applied to the actual target linkage:
  **FNDDS ↔ OFF, restricted to two blocks: yogurt and beans.** Since there is no direct ground
  truth for FNDDS↔OFF, its output is validated by plausibility spot-checks and by the diagnostics
  (block size, intra-block consistency, EM degeneracy checks) rather than precision/recall — this
  limitation is documented, not hidden.
- LLM hosting: **vLLM CPU backend** now (a modest/GPU-constrained machine won't have enough
  VRAM for an 8B model), architected behind a device-agnostic config so it's a one-variable
  switch to CUDA later (e.g. when run on a machine with an NVIDIA GPU).
- Agentic loop: **bounded autonomous loop, N=3 rounds** (configurable) for both the blocking-rule
  proposal loop and the matching-attribute proposal loop — LLM proposes → metrics computed →
  results fed back to the LLM for revision → repeat, stop early if metrics stabilize. SME review
  is represented as an exported artifact (CSV/markdown report) at the point the loop stops, not
  an interactive blocking prompt.

FDC CSV schema (verified structure, stable across releases): `food.csv` (`fdc_id`, `data_type`,
`description`, `food_category_id`) is the hub; `branded_food.csv`, `survey_fndds_food.csv`,
`sr_legacy_food.csv`, `foundation_food.csv` each key on `fdc_id` and add dataset-specific fields
(`branded_food.csv` has `gtin_upc`, `brand_owner`, `ingredients`, `branded_food_category`, etc.;
`survey_fndds_food.csv` has `wweia_category_code` joining to `wweia_food_category.csv`).
`food_nutrient.csv`, `food_portion.csv`, `food_attribute.csv` (this holds the "Additional food
description" field the vision references, via `food_attribute_type.csv`) also key on `fdc_id`.
OFF's `data/food.parquet` has `code` (barcode), `product_name` (struct array by lang),
`categories_tags`, `brands`, `ingredients_text`, `quantity`, `nutriments` (struct array).

## Project layout

```
agentic_matching/
  pyproject.toml                # uv-managed: duckdb, splink, pandas, pyarrow, httpx,
                                 # beautifulsoup4 (scrape download page), openai (vLLM client),
                                 # pydantic, typer, python-dotenv, pytest
  .env.example                  # LLM_DEVICE=cpu|cuda, LLM_MODEL, LLM_BASE_URL, VLLM_PORT, etc.
  README.md
  data/
    food.parquet                 # existing OFF data (untouched)
    raw/fdc/<dataset>/*.csv      # downloaded+extracted FDC CSVs
    parquet/fdc/<table>.parquet  # per-table Parquet conversions
    fdc.duckdb                   # persisted DB: raw tables + unified per-dataset views
    calibration/                 # gold_pairs.parquet, train.parquet, holdout.parquet
    blocks/                      # yogurt/beans candidate subsets (fndds + off sides)
    artifacts/                   # agent-loop run reports (blocking + matching), for SME review
  src/agentic_matching/
    config.py                    # paths + LLM backend config (device-agnostic)
    download_fdc.py               # scrape fdc.nal.usda.gov/download-datasets, download+unzip
                                   # only FNDDS/SR Legacy/Foundation/Branded zips
    csv_to_parquet.py             # duckdb COPY CSV -> Parquet per file
    build_fdc_db.py               # unified views joined on fdc_id per dataset type
    calibration.py                # UPC/GTIN normalization, Branded<->OFF gold pairs,
                                   # stratified sample, SME spot-check export, train/holdout split
    llm/
      client.py                   # OpenAI-compatible client wrapper against local vLLM server
      server.py                   # start/stop vLLM OpenAI-server subprocess (device from config)
      prompts.py                  # prompt templates (blocking proposal, attribute proposal,
                                   # revision-with-feedback)
    blocking/
      seed_rules.py                # vision's worked example pattern (keyword/category rules)
                                    # as the yogurt & beans starting point on both sides
      metrics.py                   # pair completeness & reduction ratio (via Branded<->OFF gold
                                    # pairs as the text-based proxy signal), block size stats
      agent_loop.py                 # bounded N=3 loop: propose -> evaluate -> revise -> stop
    attributes/
      library.py                    # versioned registry of attribute functions per block
                                     # (yogurt: is_greek/is_plain/is_drink/is_baby/
                                     #  fruit_flavored/contains_cereal/fat_level, per vision;
                                     #  beans: LLM-proposed, e.g. bean_type/is_canned/is_dried/
                                     #  is_seasoned/sodium_level)
      generator.py                   # LLM proposes attribute defs + generates extraction code
                                      # for FNDDS-side and OFF-side text fields
      correlation_check.py           # pairwise association (Cramér's V) among attributes;
                                      # flags for LLM/SME review before use in EM
    linking/
      splink_model.py                # per-block Linker (link_only, DuckDBAPI), comparisons from
                                      # attributes.library, EM training
      degeneracy_check.py            # detects collapsed/label-switched m/u estimates
      evaluate.py                    # scores against Branded<->OFF holdout (proxy) +
                                      # plausibility report for FNDDS<->OFF
      agent_loop.py                  # bounded N=3 loop: train -> score -> LLM revises attribute
                                      # set (subject to correlation_check) -> retrain
  scripts/
    01_download_fdc.py
    02_convert_parquet.py
    03_build_fdc_db.py
    04_build_calibration.py
    05_run_blocking_agent.py        # --block yogurt|beans
    06_run_matching_agent.py        # --block yogurt|beans
    07_run_splink_and_evaluate.py   # --block yogurt|beans
  tests/
    test_upc_normalization.py
    test_seed_blocking_rules.py
    test_attribute_functions.py
    test_degeneracy_check.py
```

## Build order

1. **`uv init`** the project (pyproject.toml, lockfile), add dependencies. Replace/ignore the
   pre-existing generic `.venv` (it has unrelated GPU packages, no `pyproject.toml`); `uv sync`
   will manage its own environment.
2. **`download_fdc.py`**: scrape `https://fdc.nal.usda.gov/download-datasets` for the current
   zip filenames/URLs for the four target datasets (page content changes release to release —
   don't hardcode filenames), download to `data/raw/fdc/_zips/`, unzip each into
   `data/raw/fdc/<dataset>/`.
3. **`csv_to_parquet.py`**: convert every CSV across the four extracted datasets to Parquet via
   DuckDB `COPY ... TO ... (FORMAT PARQUET)`, deduplicating shared support tables (`food.csv`,
   `nutrient.csv`, `food_nutrient.csv`, `food_category.csv`, `food_attribute.csv`,
   `food_attribute_type.csv`, `wweia_food_category.csv`, etc. appear once per dataset zip but are
   the same schema — union/dedupe by `fdc_id`/`id` while loading).
4. **`build_fdc_db.py`**: build `data/fdc.duckdb` with raw tables plus one unified view per
   dataset type (`v_fndds`, `v_sr_legacy`, `v_foundation`, `v_branded`) — each joined from
   `food.csv` + its type table + `food_category` + relevant `food_attribute` pivots (e.g.
   "Additional Description") + aggregated key nutrients — all joined on `fdc_id`. This directly
   satisfies "link everything on the USDA side using FDC ID."
5. **`calibration.py`**: normalize UPC/GTIN on both sides (strip leading zeros, handle
   UPC-A(12)/EAN-13(14) padding), inner-join `v_branded` to OFF `data/food.parquet` on normalized
   code → gold pairs. Stratify sample across `branded_food_category`/OFF `categories_tags` for
   the calibration set; export a CSV for SME spot-check; split into train/holdout
   (`data/calibration/train.parquet`, `holdout.parquet`).
6. **`llm/` module**: `config.py` reads `LLM_DEVICE` (default `cpu`); `server.py` launches
   `vllm serve <model> --device cpu ...` (model default: an ungated Llama-3.1-8B-Instruct mirror,
   overridable via `LLM_MODEL` env var) as a subprocess exposing an OpenAI-compatible endpoint on
   `localhost:$VLLM_PORT`; `client.py` is a thin `openai` SDK wrapper other modules call. Switching
   to an NVIDIA box later = set `LLM_DEVICE=cuda` (and appropriate dtype/tensor-parallel env
   vars) — no code changes elsewhere.
7. **Blocking stage** (`blocking/`): seed rules from the vision's worked pattern (FNDDS: keyword
   match on description/"Additional Description" + WWEIA category; OFF: keyword match on
   `product_name`/`categories_tags`) for yogurt and beans. `metrics.py` computes pair
   completeness & reduction ratio using the Branded↔OFF gold pairs as the calibration proxy
   (apply the FNDDS-side predicate to Branded Foods text as a stand-in, OFF-side predicate to OFF)
   plus block-size diagnostics. `agent_loop.py` runs up to 3 rounds: LLM sees current rule +
   metrics, proposes a revision, re-evaluate, stop early if pair completeness/reduction ratio
   stabilize (delta below threshold); each round's rule + metrics logged to
   `data/artifacts/blocking_<block>_round<N>.json` for SME review.
8. **Matching-attribute stage** (`attributes/`): `library.py` seeds the yogurt attributes named
   verbatim in the vision (`is_greek`, `is_plain`, `is_drink`, `is_baby`, `fruit_flavored`,
   `contains_cereal`, `fat_level`); beans attributes are LLM-proposed from scratch.
   `generator.py` samples candidate pairs within the block, asks the LLM to identify
   distinguishing patterns and emit Python extraction functions (versioned files under
   `attributes/generated/<block>/v<N>.py`, unit-tested). `correlation_check.py` computes Cramér's
   V between attributes on the block population and flags highly-correlated pairs before they're
   handed to EM.
9. **Linking stage** (`linking/`): per block, build a splink `Linker` (`link_only`, two input
   frames — FNDDS block subset & OFF block subset, `DuckDBAPI`), comparisons built from the
   attribute library + `gtin_upc`-style term-frequency-adjusted exact-ish fields where available,
   run `estimate_parameters_using_expectation_maximisation`. `degeneracy_check.py` inspects
   fitted m/u probabilities for collapse/label-switching before accepting a run.
   `evaluate.py` scores against the Branded↔OFF holdout proxy (precision/recall/F-measure) and
   produces a plausibility report (score distribution, top/bottom-N examples) for FNDDS↔OFF since
   no direct ground truth exists there. `agent_loop.py` runs up to 3 rounds: LLM sees eval
   results, proposes attribute-set adjustments (subject to the correlation check), re-trains,
   re-scores, stop early on stabilization — artifacts logged the same way as the blocking loop.
10. **`scripts/0N_*.py`** are thin CLI entry points (typer) chaining the above, runnable
    independently or as `uv run scripts/0N_*.py`.
11. **Tests**: UPC normalization edge cases, seed blocking rule membership on hand-built
    fixtures, attribute-function unit tests, degeneracy-check detection on a synthetic
    label-switched fit.

## Verification

- `uv run scripts/01_download_fdc.py` → confirm 4 dataset dirs populated under `data/raw/fdc/`.
- `uv run scripts/02_convert_parquet.py` then `duckdb` query row counts match source CSVs.
- `uv run scripts/03_build_fdc_db.py` → spot-check `v_fndds`/`v_branded`/etc. row counts and a
  join sample against known `fdc_id`s.
- `uv run scripts/04_build_calibration.py` → report gold-pair count, category coverage, and
  train/holdout sizes; open the SME spot-check CSV.
- Start the vLLM CPU server (`uv run python -m agentic_matching.llm.server`) and confirm a
  simple prompt round-trips via `client.py`.
- `uv run scripts/05_run_blocking_agent.py --block yogurt` and `--block beans` → inspect
  `data/artifacts/blocking_*` for the metric trajectory across rounds.
- `uv run scripts/06_run_matching_agent.py --block yogurt|beans` → inspect generated attribute
  code + correlation flags.
- `uv run scripts/07_run_splink_and_evaluate.py --block yogurt|beans` → check EM converged
  (no degeneracy flag), review precision/recall on the Branded↔OFF proxy holdout and the
  FNDDS↔OFF plausibility report.
- `uv run pytest`.
