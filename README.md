# Agentic Food Data Linkages Constructor — POC

This tool links USDA FoodData Central (FDC) records to a food-product database. Today
that database is Open Food Facts (OFF). The tool uses `splink` (DuckDB backend) to do
the probabilistic matching. A small LLM proposes and revises the blocking rules and the
matching attributes. The tool mirrors the manual process a subject-matter expert (SME)
uses today. See `PLAN.md` for the full design.

FNDDS is the fixed side in every stage of this pipeline. The other side is a pluggable
`CatalogSource` (`catalog_source.py`). See "The pluggable second side" below. Open Food
Facts is the one real `CatalogSource` in use today. Because of this, the terms
`off_*` and `catalog_*` mean the same thing throughout this document and the code. The
`catalog_*` names are the general term. "OFF" is what that term means right now.

## Setup

```bash
uv sync
cp .env.example .env   # adjust if needed; sane CPU defaults are baked in
```

You must have `data/food.parquet` (the Open Food Facts export) before you run the
pipeline. No script here downloads it for you. Download it once yourself:

```bash
curl -L -o data/food.parquet \
  https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet
```

This file is the full Open Food Facts product export, in parquet format (about
7.7GB). The download takes a while. Get the newest version from the
[Hugging Face dataset page](https://huggingface.co/datasets/openfoodfacts/product-database)
if this URL changes.

## The pluggable second side (`catalog_source.py`)

Every stage of the pipeline — blocking, matching attributes, linking, profiling,
calibration — reads the non-FNDDS side's settings from one `CatalogSource` value. This
value is `catalog_source.py`'s `ACTIVE_CATALOG_SOURCE`. No stage hardcodes Open Food
Facts' specific column names or file shape. `CatalogSource` is a plain, frozen Python
dataclass:

```python
CatalogSource(
    name="off", display_name="Open Food Facts (OFF)",
    raw_parquet=OFF_PARQUET, search_text_parquet=OFF_SEARCH_TEXT_PARQUET,
    id_col="code", product_name_col="product_name", category_col="categories_tags",
    category_kind="array_contains",  # OFF's categories_tags is an array of tags
    brand_col="brands", search_text_col="search_text",
    generic_term_min_doc_count=None,  # derived from row count -- see profiling.py
    flatten_sql=off_text.flatten,     # OFF's struct/array -> flat-text preprocessing
)
```

To point the pipeline at a different food-product database — for example, a retail
catalog such as Circana, which PLAN.md names as OFF's original stand-in — build a new
`CatalogSource` for it. Give it its own paths and column names. Set `category_kind` to
`"exact"` if the category field holds one plain value, the way FNDDS's own WWEIA
category does. Set it to `"array_contains"` if the category field holds a list of tags,
the way OFF's does. Write a `flatten_sql` function if the raw data needs work before it
has a flat `search_text` column, the way OFF's struct/array fields do. Then point
`ACTIVE_CATALOG_SOURCE` at the new value. No blocking, attributes, linking, or
profiling code needs to change.

`tests/test_catalog_source_abstraction.py` checks this claim. It builds a second,
fake `CatalogSource` with different column names and `"exact"` category matching, and
runs real code against it. This proves the abstraction works for more than just OFF
under a new name.

FNDDS itself has no `CatalogSource`. This is a deliberate choice, not an oversight.
FNDDS is the fixed side in every real use case this project targets. It has about
5,400 rows. It needs no struct/array flattening step and no scale-tuned constants. The
other side is the one expected to change. See `catalog_source.py`'s module docstring
for the full reasoning.

Elsewhere in this document, the names `off_*` and `catalog_*` mean the same active
`CatalogSource`, since OFF is that source today. Any OFF-specific column name or number
below describes the *current* deployment. It is not a fixed assumption in the code.

## Pipeline

```bash
uv run scripts/01_download_fdc.py          # scrape+download FNDDS/SR Legacy/Foundation/Branded
uv run scripts/02_convert_parquet.py       # CSV -> Parquet, + flatten OFF text fields
uv run scripts/03_build_fdc_db.py          # data/fdc.duckdb: unified per-dataset views, + profiling
uv run scripts/04_build_calibration.py     # Branded<->OFF gold pairs, train/holdout split
uv run scripts/05_run_blocking_agent.py --block yogurt   # (or beans)
uv run scripts/06_run_matching_agent.py --block yogurt
uv run scripts/07_run_splink_and_evaluate.py --block yogurt
uv run scripts/08_visualize_matches.py waterfall --block yogurt   # or: weights | histogram | dashboard

# Or run 05-07 as one bounded outer loop. It re-blocks automatically if linking's
# result looks like a blocking problem, not just an attribute problem (see below):
uv run scripts/09_run_outer_loop.py --block yogurt
```

Each agent-loop script writes a log of every round to `data/artifacts/<block>/<run_id>/`.
This log is for SME review. Each top-level run gets its own timestamped directory (see
`config.py`'s `new_run_id`/`run_artifacts_dir`). Because of this, one run of a block
never overwrites another, and you can compare runs over time. Inside a run's directory:

- `blocking_round<N>.json` and `attributes_round<N>.json` hold the proposed rule or
  attributes, plus their metrics.
- `linking_round<N>.json` holds degeneracy flags, holdout evaluation results, and a
  plausibility summary (the score distribution, plus the top 10 and bottom 10 examples).
- `matches_round<N>.csv` holds every predicted FNDDS↔OFF pair from that round, not just
  the JSON's top/bottom 10. Rows are sorted by `match_probability`, highest first. Each
  matching attribute's value is shown for both sides. Open this file directly in a
  spreadsheet to check the real match results.

`final_matches.csv` has no round number. Each round overwrites it. It always holds the
result from the *best* round this run produced, not necessarily the last round (see
"A later round can regress" below). This file is the real deliverable.

`linking/evaluate.py::best_match_per_catalog` builds this file. For each catalog
record, it keeps only the single best (highest-`match_probability`) FNDDS match. The
goal is to attach nutrition information from FNDDS to each commercial product. Each
product should end up with one nutrition profile, not several competing FNDDS
candidates. One FNDDS record can still match many different commercial products — for
example, many brands' "Black Beans" can all point at the same "Black beans, canned"
nutrition profile. Only the catalog side gets this one-best-match rule.

This deduplication step does not assume OFF. `best_match_per_catalog` only needs a
`unique_id_r` column that identifies the commercial-product side. The same pipeline
works, unchanged, against whatever `ACTIVE_CATALOG_SOURCE` points to (see "The
pluggable second side" above) — for example, a retail catalog such as Circana, in
place of OFF.

### A later round can regress: the loop keeps the best round, not the last one

Each of the three agent loops — blocking, attributes, linking — can run for several
rounds before it stops (either it stabilizes, or it hits `max_rounds`). A later round
is not always better than an earlier one. An LLM revision can make a rule or attribute
set worse. Nothing stops this unless a check looks for it directly. Because of this,
each loop picks its final result from *all* completed rounds, using whatever quality
signal it has. It does not simply use the last round it ran.

- `blocking/agent_loop.py::_select_final_rule` favors an earlier round when the change
  from the second-to-last round to the last round was small (see its docstring for a
  real case: a revision grew a block from 148 to 1,282 FNDDS records while the proxy
  metric moved by less than the stabilization threshold). It also compares every round
  against the hand-written seed rule for that block, when one exists, and keeps the
  seed rule if no round beats it. This guards against a first LLM round that is worse
  than the seed rule it was meant to improve on.
- `attributes/agent_loop.py::select_final_attributes` favors the round with the fewest
  correlation flags. This is the only quality signal this loop has, since it never
  trains a real matching model.
- `linking/agent_loop.py::select_best_round` favors the round with the highest holdout
  F1 score. It rejects any round with zero confident real-world matches outright, no
  matter how good that round's calibration-proxy F1 score looks. Real case (yogurt
  block, model `qwen3:8b`): round 3 reached F1=0.048 with 55,516 confident matches.
  Round 4's revision dropped real-world matching to **zero** confident matches out of
  296,000 candidates. Round 4's holdout F1 (0.024) looked only like a return to round
  0's baseline — not an obvious disaster from that number alone. When the loop's last
  round is not the best round, it rebuilds `final_matches.csv` from the best round's
  attributes before it returns. This step is cheap: splink training and prediction take
  seconds; the LLM call is the expensive part. `outer_loop.py`'s own reported metrics,
  and its `diagnose_blocking_problem` check, use this same best-round selection. They
  do not use the last linking round by default.

## Outer loop: blocking↔linking feedback (`scripts/09_run_outer_loop.py`)

`linking/agent_loop.py`'s inner loop only revises the *attribute* set. It never
revises the *blocking* rule that chose which records were candidates in the first
place. `outer_loop.py` closes this gap. It runs blocking, then attributes, then
linking, once. Then `diagnose_blocking_problem` checks the last linking round for two
signs that point at the blocking rule, not the attributes (an attribute-shaped problem
is already the inner loop's job to fix):

1. Too few raw candidate pairs (`n_candidate_pairs` under `outer_loop.MIN_CANDIDATE_PAIRS`).
2. A `collapsed` degeneracy flag that survives every round of attribute revision.

If either sign appears, the outer loop sends this finding back into a new blocking
round (as `prior_linking_findings`, through `run_blocking_agent`'s `linking_feedback`
parameter). Then it runs the whole pipeline again. This retry is bounded by
`AGENT_MAX_OUTER_ROUNDS` (default: 2). The default gives re-blocking one chance to fix
the problem. It is not an open-ended search.

Each outer round writes to `data/artifacts/<block>/<run_id>/outer_loop_round<N>.json`.
This is the same run directory the blocking, attributes, and linking stages write to
(see above). One call to `run_outer_loop` uses one shared run directory across every
stage it runs.

`--steps` runs only some of the three stages, against a block that already went
partway through the pipeline. For example:

```bash
# Redo attribute selection (and relink) without touching the existing blocking rule:
uv run scripts/09_run_outer_loop.py --block beans --steps attributes,linking

# Just relink against whatever attributes are already saved:
uv run scripts/09_run_outer_loop.py --block beans --steps linking
```

A skipped stage does not rerun with cached results. The tool simply does not touch it.
Any later stage you did include uses that stage's last saved output as-is. The tool
still copies that skipped stage's diagnostic files forward from the most recent
earlier run, into this run's directory. Because of this, even a `--steps linking` run's
folder looks like a complete record — it has blocking and attribute files too, just
carried over, not freshly made. This avoids leaving files silently missing.

If no earlier run has the files to copy — this is the first run for a block, or the
earlier run also skipped that stage — the tool fails loudly with a `FileNotFoundError`.
It does not silently leave the new run's folder incomplete. In that case, run the
pipeline once with the stage included, first.

The re-blocking feedback step above fires only when `--steps` includes both `blocking`
and `linking`. With no `linking` step, there is no later result to re-block in
response to. If `blocking` or `linking` is missing, the tool runs a single round and
logs a warning if it finds a blocking-shaped problem it cannot act on this run.

## Visualizing matches (`scripts/08_visualize_matches.py`)

This script wraps splink's own Altair/HTML chart methods (`linking/charts.py`). It
uses a block's *current* attribute set (`attributes/generated/<block>/latest.json`)
and retrains a fresh model each time it runs, since this pipeline never saves a
trained model to disk (the same is true of `linking/evaluate.py`). Each chart is
written next to that block's most recent pipeline run, at
`data/artifacts/<block>/<run_id>/chart_<kind>.html` — open it directly in a browser.

- `waterfall --block yogurt [--n 10] [--mode stratified|top|bottom|borderline] [--threshold 0.0]`
  — shows how each comparison added to the final match score, for a set of pairs.
  `stratified` (the default) spreads the chosen pairs evenly across the whole score
  range. This gives a mix of clear matches, clear non-matches, and borderline cases,
  instead of `n` pairs that all look alike.
- `weights --block yogurt` — shows the model's learned strength of evidence for each
  comparison level. This chart needs no predictions, only the trained model.
- `histogram --block yogurt [--threshold 0.0]` — shows the spread of match weights
  across every predicted pair in the block.
- `dashboard --block yogurt [--num-example-rows 3]` — splink's interactive
  comparison-viewer. This is the most thorough option for manual SME review. The
  tradeoff is a larger HTML file, a few MB in size.

The waterfall chart needs splink's `retain_intermediate_calculation_columns=True`
setting. The main training path (`linking/splink_model.py::build_linker`) sets this to
`False` by default (see "Known constraints" below). `charts.py` sets it to `True`
directly, for its own, separately trained linker. This is safe at this project's block
sizes (checked: about 3.3GB peak memory for yogurt's roughly 660,000 candidate pairs).
The real past cause of high memory use was *unbounded EM blocking*, fixed in
`train()`. It was not this flag on its own.

## Corpus profiling: ground LLM proposals in real catalog statistics

`profiling.py` computes exact word-frequency counts and category-field distributions,
once, over the **full** FNDDS and OFF datasets (as part of `scripts/03_build_fdc_db.py`).
This step is fast even at OFF's roughly 4.66 million rows — DuckDB's vectorized
execution finishes it in under a second, so no sampling step is needed. The results go
to `data/profiling/`.

Without this step, the blocking and matching-attribute agent loops only saw a few
dozen sample records per round. This is enough to find plausible-looking keywords or
categories, but gives no way to tell "specific to this block" apart from "common
across the whole catalog." A keyword such as `protein`, `rice`, or `black` looks safe
in a 40-row yogurt or beans sample. But on its own, each one matches 20,000 to 65,000
unrelated OFF records across the whole catalog. Two parts of the pipeline now use the
precomputed statistics instead of guessing:

- **Blocking** (`blocking/agent_loop.py`): each round's prompt includes `corpus_stats`
  — each side's catalog size, plus its most common catalog-wide terms. The system
  prompt tells the LLM not to propose any of these terms as a standalone keyword,
  unless the term is the block's own name. For the catalog side, `llm/mock.py` also
  *enforces* this rule as a hard rejection, instead of only suggesting it, since the
  mock cannot judge keyword breadth the way a real LLM can. The rejection threshold is
  `profiling.generic_term_min_doc_count("catalog")`. This is a *fraction* of catalog
  size, not a fixed number. Its value is set to match OFF's original, checked
  threshold of 15,000 matches, at OFF's real scale of about 4.66 million rows. Because
  it is a fraction, it stays meaningful if a different catalog has a different row
  count. There is no matching hard rejection for FNDDS. FNDDS has only about 5,400
  rows total, so even a broad FNDDS keyword is not a memory risk. Applying the same
  rule to FNDDS would only throw away good keywords (for example, "cooked" or
  "canned") for no safety gain.
- **Matching attributes** (`attributes/agent_loop.py`): each round's prompt includes
  `field_stats` — the most common real values of each side's category fields (OFF's
  `categories_tags` and `brands`; FNDDS's WWEIA category), counted only within this
  block's own records. A proposed category attribute (for example, `bean_type`'s
  categories) can then be based on values that really occur, instead of values
  invented from general world knowledge.

`data/profiling/` holds derived data. Like `data/blocks/`, it is not checked into git.
Rebuild it with `uv run scripts/03_build_fdc_db.py`, or with
`uv run python -m agentic_matching.profiling`.

## Structured category-based blocking: not just free-text keywords

Free-text keyword matching against FNDDS's blob of description, WWEIA category, and
"Additional Description" text turned out to be easy to trip up. FNDDS's
`additional_description` field is full of stock phrases ("all flavors", "multigrain,
whole grain, whole wheat") shared across many unrelated food categories. Even a
keyword taken from clean description text (for example, "fruit", "whole", "plain")
often occurs, on its own, in many other foods' descriptions too (fruit salad, whole
wheat muffins, plain pretzels). Real case, from this project's own data: the yogurt
block's FNDDS side held **1,176 records, and 1,114 of them (95%) were not yogurt at
all** — chicken, coffee, pasta, tea, sandwiches. All of these were pulled in because
"flavors," "fruit," "plain," or "whole" matched stock annotation text with nothing to
do with the food's real identity. OFF showed the same failure from a single mined
keyword: "almond" alone pulled in "Milk Chocolate With Caramelized Almonds."

Both datasets carry clean, human-written category labels for exactly this kind of
problem: FNDDS's WWEIA food category (for example, "Yogurt, regular" or "Yogurt,
Greek"), and OFF's `categories_tags` (for example, `en:yogurts`). Because of this, a
blocking rule can now name `"categories"` for each side (`blocking/rules.py`), joined
with the keyword predicate. FNDDS always matches by exact equality on
`wweia_food_category_description` (`category_kind="exact"`). The catalog side matches
using whatever kind `ACTIVE_CATALOG_SOURCE.category_kind` names — OFF's
`categories_tags` uses `"array_contains"`, since it holds a list of tags, not one
value. `blocking/agent_loop.py::_category_options` finds the real category values seen
among records already likely to belong to the block. This uses the same seed-term
filter as the keyword-mining samples. These real values are shown to the LLM (or the
mock) as `category_options`, so it picks from real values instead of inventing
category names. The FNDDS-side keyword match is now also limited to the raw
`description` column only, not the stock-phrase-heavy blob. This is a second,
separate fix.

`llm/mock.py` takes this one step further. When it finds a clean matching category for
a side, it **skips keyword guessing entirely** for that side. It proposes only the
seed vocabulary plus the category. The mock has no way to tell a bad general word
("whole," which matches unrelated foods) apart from a good, block-specific word
("greek"), the way a real LLM's world knowledge could. So when a reliable structured
signal exists, the mock trusts it over guessing. Result, on this project's real data:
yogurt's FNDDS side went from 1,176 records (95% wrong) to **61 records, 0 wrong**.
Beans went from 908 records to 188. The 26 records in that beans block with no "bean"
word in them are real legume foods (chickpeas, lentils, split peas), correctly kept by
the "Beans, peas, legumes" WWEIA category. This is real recall, not noise. A real LLM
should be able to keep mining keywords selectively alongside categories — telling
apart a block-specific mined word from a general one — instead of switching off
keyword mining entirely. The mock's all-or-nothing rule is a known simplification.

Note: the calibration-proxy metrics (pair completeness) cannot fully show this gain.
Branded Foods (the FNDDS-side text stand-in used for calibration, see below) has no
WWEIA-style category field, so the FNDDS-side category rule cannot be checked against
that proxy (`blocking/metrics.py::pair_completeness` passes `category_col=None` for
FNDDS for this reason). The precision gain described above was confirmed by looking
directly at the materialized block, not by the automatic metric. This is exactly the
kind of gap the "plausibility spot-check" review step (see `PLAN.md`) exists to catch.

## Mining candidate matching attributes from the block itself

For a from-scratch block (one with no `seed_rules.SEED_ATTRIBUTES` entry, for example
`beans`), the first round used to propose attributes purely from general knowledge —
or, for the mock, from a hand-written list. Nothing actually *chose* attributes from
the real data. `attributes/agent_loop.py::_candidate_boolean_terms` now mines the
block's own free text (FNDDS `description`, catalog `search_text`) for words that
split the block's records into a real minority and majority, on at least one side (a
`min_frac`-to-`max_frac` band — a word near 0% or near 100% tells you nothing). Terms
are ranked by the *lower* of the two sides' match fractions. This way, a term with
real signal on both sides (a true cross-dataset concept, for example "meat") ranks
above a term with signal on only one side. (FNDDS's beans block also catches some
unrelated "mixed dish" records — pasta, potato, and sandwich dishes that happen to
mention beans. Their own vocabulary would otherwise dominate a plain frequency count.)
Every round's prompt includes this list, as `candidate_terms`.

`llm/mock.py` now uses this list instead of a hand-written boolean attribute list. For
a from-scratch block, it combines a small hand-written *category* exception list
(`_CATEGORICAL_EXCEPTIONS` — today, only `beans`' `bean_type` and `sodium_level`) with
boolean attributes built directly from the top mined `candidate_terms` (for example,
`has_meat`, `has_canned`). The exception list stays hand-written because grouping
synonyms across languages — "garbanzo," "chickpea," "pois chiche" — into one category
needs reasoning that plain word-counting cannot do. A later round's `metrics.py` check
still catches a mined attribute that turns out to be redundant, the same way it would
catch one the LLM proposed directly. Real case: mining once surfaced `has_black`,
which repeats `bean_type`'s "black" category. The check flagged it at a Cramér's V of
0.972 and dropped it in round 1, with no extra prompting needed.

**Real limits worth knowing:** a single mined word only catches that exact word.
`has_meat`, when mined, fires only on the word "meat" itself. A hand-written
`has_meat` attribute could list "pork," "beef," "bacon," "sausage," and "ham" as
synonyms, and so catch far more records. Mining also cannot guarantee that every
useful attribute is found. In this project's real data, the word `rice` ranked about
16th (just outside the mock's top-6 cutoff), because its signal on the OFF side of
this block was weaker (about 1.2%) than `meat`'s (about 5.3%). Because of this,
`with_rice` does not currently appear from mining alone. **This is exactly the kind of
gap a real LLM should close** (see below). A real LLM is not limited to counting exact
word matches. It can reason that "rice matters for a beans-and-rice mixed dish," the
same way it can reason that "porc" means "pork."

### What changes when a real LLM replaces the mock?

**Nothing else in the code.** `corpus_stats`, `field_stats`, and `candidate_terms` are
built by `blocking/agent_loop.py` and `attributes/agent_loop.py`. They are sent to
`ChatClient.complete_json` as plain prompt text (see `llm/prompts.py`). The same
payload reaches `llm/mock.py` and a real Ollama-backed `llm/client.py` model in the
same form. Switching between them is a one-variable change (see "LLM backend" below).
No prompt, agent-loop, correlation-check, or splink code needs to change. What *does*
change is quality. A real LLM sees the same `candidate_terms` grounding, but it is not
limited to counting exact words. It can propose `with_rice` even though "rice" ranked
in the middle of the list. It can recognize that "pork," "beef," and "bacon" all
belong under `has_meat`. It can recognize matching words across languages ("porc,"
"riz"), which the mock's own docs say it cannot do.

## LLM backend

Every agent-loop script calls `get_llm_client()`. The `LLM_DEVICE` setting picks the
backend.

- **`LLM_DEVICE=ollama` (the default)**: talks to a local or remote
  [Ollama](https://ollama.com/download) server (one installer, no manual build steps)
  over its OpenAI-compatible `/v1` API. Run
  `uv run python -m agentic_matching.llm.server` in its own terminal, before
  `scripts/05` through `09`. This starts the server, or connects to one already
  running. The scripts only ever *talk* to a server; they never start one themselves.
  `llm/server.py::OllamaServerManager` handles two things worth knowing. First, Ollama
  is often already running as a background service (the official installer sets this
  up on Linux). The manager checks for a server that already answers, and uses it
  instead of erroring out or starting a second one. It only starts its own `ollama
  serve` process if nothing answers yet, and it only stops a server it started itself.
  Second, Ollama needs a model *pulled* before it can serve it, so `start()` runs
  `ollama pull <model>` for you. This is a fast no-op if the model is already present.
  The defaults are `LLM_MODEL=qwen2.5:1.5b` and `LLM_PORT=11434` (Ollama's own
  defaults); override them in `.env` if you need to. You need a fairly recent Ollama
  version for its OpenAI-compatible `/v1` API, including `response_format` JSON-mode
  support, which `llm/client.py` needs. This was checked against a real Ollama
  install during this project's development.
- **`LLM_DEVICE=databricks`**: a Databricks Model Serving pay-per-token endpoint.
  Credentials come from the same `DATABRICKS_HOST` and `DATABRICKS_TOKEN` environment
  variable names the Databricks CLI and SDK use — not a separate, `LLM_`-prefixed
  copy. `DATABRICKS_LLM_ENDPOINT` can be either a bare serving-endpoint name, or the
  full invocations URL you can copy from the Databricks UI's "Query endpoint" page
  (see `.env.example`). There is nothing to start or manage; it already runs as a
  cloud service. Every call is a direct HTTP POST — not the `openai` SDK's normal URL
  building — to that endpoint's own literal
  `.../serving-endpoints/<name>/invocations` URL. See `llm/client.py`'s module
  docstring for why: Databricks' documented shared-gateway pattern (routing by a
  `model` field in the request body) returned an HTML login-page redirect for a real
  named endpoint on a real workspace, while the literal per-endpoint URL returned a
  real API response. The token also needs `model-serving` or
  `model-serving-inference` scope, which a scoped OAuth or service-principal token may
  not have by default, even though a full personal access token usually does. This was
  checked end-to-end against a real endpoint
  (`databricks-meta-llama-3-3-70b-instruct`). **This is a paid endpoint.** Every LLM
  call in every agent loop costs real money while `LLM_DEVICE=databricks` is set, and
  a run makes many calls. Switch back to `LLM_DEVICE=ollama` when you are not using it
  on purpose.
- `LLM_DEVICE=mock`: works offline, with no server needed. `llm/mock.py` implements
  the same `ChatClient` interface, using fixed keyword-mining rules. This is enough to
  run, test, or demo the full pipeline end to end, with no LLM installed.
- Set `LLM_BASE_URL` instead, to point at a server that is already running (a remote
  host, or a separately managed Ollama instance), instead of starting one locally.
  This setting does not apply to `LLM_DEVICE=databricks`, which always uses its own
  literal invocations URL (see above).

## Known limits at this project's data scale

The pipeline filters each block down to its own small subset of records first
(`data/blocks/<block>_{fndds,catalog}.parquet`, written once by
`scripts/05_run_blocking_agent.py`), before splink ever runs. Because of this, splink
never touches the full roughly 4.66-million-row OFF table directly. Even so, each
block is still uneven in size: hundreds to a few thousand FNDDS records, against tens
of thousands of catalog records. A few things below turned out to matter at that
scale. All are fixed today, but are worth knowing if this pipeline grows (larger
blocks, more attributes, and so on). Check these first on a machine with limited
memory:

- **EM blocking on one skewed attribute.** `linking/splink_model.py`'s EM training
  passes always combine an attribute column with the search-text prefix condition
  (`block_on(col, "substr(l.search_text, 1, 4)")`). They never block on the attribute
  alone. Blocking on a skewed boolean attribute alone (for example, `is_greek`, which
  is False for almost every record) pairs up tens of millions of records. This was the
  original cause of memory exhaustion during development.
- **Uncapped, exhaustive holdout scoring.** `linking/evaluate.py`'s holdout scoring
  uses an exhaustive (`1=1`) blocking rule. This is safe only because the holdout
  sample size has a cap (`max_holdout_positives`, default 500). Branded Foods
  publishes the same product many times, once per row, under the same GTIN. Without
  the cap, an uncapped category holdout can reach tens of thousands of rows, and an
  exhaustive cross join at that size was the other original cause of memory
  exhaustion.
- **Keyword-mining samples used to be nondeterministic.**
  `blocking/agent_loop.py::_sample_texts` now sorts explicitly, by `fdc_id` or `code`.
  Without this sort, `LIMIT` alone left row order — and so which records got sampled
  for keyword mining, by the mock or by a real LLM — up to DuckDB's own query plan.
  This was seen to swing a block's catalog-side size by 2 to 3 times, across runs of
  the exact same code.
- **Mined keywords used to be too broad.** A plain stopword list does not solve this:
  new general words such as `protein`, `rice`, `black`, and `green` kept slipping
  through, and each one independently matched 20,000 to 65,000 OFF records on its own.
  A plain *relative* frequency threshold is also too loose at OFF's roughly
  4.66-million-row scale, since even a "rare" word, by percentage, can still match a
  large number of records. See "Corpus profiling" above for the real fix:
  `profiling.py`, the `corpus_stats` field in the prompt, and a hard rejection rule in
  `llm/mock.py` for the catalog side. This rejection rule only *forces* the outcome
  for the mock. A real LLM should judge keyword breadth on its own, from the same
  `corpus_stats`. But if a proposed rule still produces an unusually large block,
  check `data/blocks/<block>_catalog.parquet`'s row count before you run the linking
  stage.

If `scripts/07_...` runs for a long time, or memory use keeps climbing, stop it and
check `data/blocks/<block>_catalog.parquet`'s row count. A much larger catalog-side
block than the ones checked here may need a tighter blocking rule before you run EM
training.

**TODO: the linking holdout's exclusion rule only uses the seed rule, not the block's
real, current rule.** `linking/evaluate.py::_load_block_holdout` removes known
false-positive categories from the calibration holdout, using the `exclude_keywords`
field from `blocking/seed_rules.json`. A block with no seed entry — or a seed entry
whose excludes have not caught up with what the block's real, current rule has since
learned — silently gets less exclusion than its real rule would give it. Real case:
`beans` had no seed entry at all for most of this project's history. As a result, 162
real jelly-bean-candy rows (`branded_food_category=Candy`, OFF tag `en:jelly-beans`)
polluted its holdout F1 score and its other holdout-based checks. The fix used so far:
add and extend the `beans` seed entry with the excludes its own round-1 blocking rule
had already found on its own (`jelly`, `vanilla bean`, `protein powder`, `crisps`).
This is a fix for one block, not a fix for the whole system. Any future block with no
hand-written seed — or one whose seed goes stale relative to its real rule — will hit
the same silent gap. A more complete fix would build the holdout excludes from the
block's real, current, best rule, instead of (or in addition to) the seed. This fix
was deliberately left for later: that real rule is itself LLM-proposed, so using it to
define the ground truth the same LLM's attribute revisions get scored against risks a
smaller version of the same circularity problem `blocking/metrics.py::
term_predicate_sql`'s docstring already avoids, on the inclusion side. (An
over-aggressive learned exclude list could shrink the holdout down toward the cases
the current attributes already handle well, quietly inflating holdout F1 with the
model's own past choices.) This fix would also need a saved, canonical "final rule"
file for blocking, which does not exist today — unlike attributes'
`generated/<block>/latest.json`. `blocking/agent_loop.py` picks a final rule in
memory, and never saves it separately from the per-round files.

## Calibration data quality: description similarity check

`calibration.py` builds the Branded↔OFF gold-pair set by matching normalized UPC/GTIN
codes only. A shared barcode is strong evidence of a real match, but not proof —
barcode reuse, a relisted product, or a data-entry error on either side could still
produce a "gold" pair whose two descriptions do not actually agree.

To make this risk visible, every gold pair now carries a `description_similarity`
score: a word-level Jaccard similarity between the Branded description and the OFF
product name, computed directly in DuckDB (see `calibration.py::
_description_similarity_sql`). This score is **not** used to filter the calibration
set automatically. Checked directly against this project's real data (1,777,551 gold
pairs; mean similarity 0.78, median 1.0), the small tail of zero-similarity pairs
(40,741 pairs, 2.3%) turned out to be a mix, not mostly bad data: real matches in two
languages, a real brand name that shares no words with a generic description, and
Branded descriptions that name only the packaging ("Aluminum Cans") rather than the
product. A blind cutoff would have thrown out real matches along with bad ones. So the
score is only a signal for human review, surfaced two ways:

- `data/calibration/sme_spot_check_sample.csv` — the existing stratified SME sample,
  now including each pair's `description_similarity`.
- `data/calibration/low_similarity_examples.csv` — the 200 gold pairs with the lowest
  `description_similarity` score, worst first, written by
  `calibration.py::export_low_similarity_examples`. This file exists for a human to
  check the real riskiest-looking pairs directly, rather than a random sample.

## Testing

```bash
uv run pytest
```
