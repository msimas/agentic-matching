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

Requires `data/food.parquet` (Open Food Facts) to already be present — it isn't
fetched by any script here, so download it once yourself:

```bash
curl -L -o data/food.parquet \
  https://huggingface.co/datasets/openfoodfacts/product-database/resolve/main/food.parquet
```

This is Open Food Facts' full product export in parquet form (~7.7GB), so the
download takes a while; get the latest version straight from the
[Hugging Face dataset page](https://huggingface.co/datasets/openfoodfacts/product-database)
if this URL ever moves.

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

# Or run 05-07 as one bounded outer loop that re-blocks automatically if linking's
# result looks like a blocking problem, not just an attribute problem (see below):
uv run scripts/09_run_outer_loop.py --block yogurt
```

Each agent-loop script logs every round to `data/artifacts/<block>/<run_id>/` for SME
review, one timestamped directory per top-level invocation (see config.py's
`new_run_id`/`run_artifacts_dir`) so successive runs of the same block don't overwrite
each other and can be compared over time. Within a run's directory:
`blocking_round<N>.json` and `attributes_round<N>.json` hold the proposed
rule/attributes plus their metrics, and `linking_round<N>.json` holds degeneracy flags,
holdout evaluation, and a plausibility summary (score distribution, top/bottom-10
examples). For the linking stage specifically, `scripts/07_...` also writes every
predicted FNDDS<->OFF pair (not just the JSON's top/bottom-10) to
**`matches_round<N>.csv`** — sorted by `match_probability` descending, with each
matching attribute's value on both sides alongside it — open that directly in a
spreadsheet to inspect the actual match results.

**`final_matches.csv`** (no round number — overwritten each round, reflects the *best*
round the loop produced this run, not necessarily the last one; see `select_best_round`
below) is the actual deliverable, distinct from the review CSV above:
`linking/evaluate.py::best_match_per_off` collapses every candidate pair down to
the single best (highest `match_probability`) FNDDS record per OFF/commercial-product
record. The real goal here is attaching nutritional information (FNDDS) to commercial
products — a product should end up with *one* nutrition profile attached, not several
competing FNDDS candidates — while one FNDDS record can legitimately attach to many
different commercial products (many brands' "Black Beans" can all point at the same
"Black beans, canned" nutrition profile), so only the OFF side is deduplicated. Nothing
about this assumes OFF specifically: it only needs a `unique_id_r` column identifying
the commercial-product side, so the same pipeline would work unchanged against a
proprietary retail catalog (e.g. Circana) substituted for OFF.

### A later round can regress -- picking the best round, not just the last one

All three agent loops (blocking, attributes, linking) can run for several rounds before
stopping (stabilization or `max_rounds`), and a later round isn't guaranteed to be
better than an earlier one -- an LLM revision can make things worse, and the loop has
no built-in reason to notice unless something explicitly checks. Each loop now selects
its final result from *all* completed rounds using whatever quality signal it actually
has, rather than blindly using whichever round happened to run last:

- `blocking/agent_loop.py::_select_final_rule` -- prefers an earlier round if the
  *last* round-to-round change in pair completeness/reduction ratio was negligible
  (see its docstring for the verified case: a revision that grew the block from 148 to
  1,282 FNDDS records while the proxy metric moved by less than the stabilization
  threshold).
- `attributes/agent_loop.py::select_final_attributes` -- prefers the round with the
  fewest correlation flags (this loop's only quality signal, since it never trains a
  real model).
- `linking/agent_loop.py::select_best_round` -- prefers the round with the highest
  holdout f1, but disqualifies any round with zero confident real-world matches
  outright regardless of how good its calibration-proxy f1 looks. Verified real case
  (yogurt, `qwen3:8b`): round 3 reached f1=0.048 with 55,516 confident matches; round
  4's revision collapsed real-world matching to **zero** confident matches out of 296K
  candidates, while its holdout f1 (0.024) merely looked "back to round 0's baseline" —
  not obviously catastrophic on that number alone. If the loop's last round isn't the
  selected best one, `final_matches.csv` is regenerated from the best round's
  attributes before the loop returns (cheap: splink training/prediction is seconds,
  the LLM call is what's expensive). `outer_loop.py`'s own reported metrics and
  `diagnose_blocking_problem` use the same selection, not the last linking round.

## Outer loop: blocking<->linking feedback (`scripts/09_run_outer_loop.py`)

`linking/agent_loop.py`'s inner loop only ever revises the *attribute* set — it never
revises the *blocking* rule that determined which records were candidates in the first
place. `outer_loop.py` closes that gap: it runs blocking → attributes → linking once,
then `diagnose_blocking_problem` inspects the last linking round for two symptoms that
specifically implicate the blocking rule rather than the attributes (attribute-shaped
weaknesses are already the inner loop's job to fix): too few raw candidate pairs
(`n_candidate_pairs` under `outer_loop.MIN_CANDIDATE_PAIRS`), or a `collapsed`
degeneracy flag surviving every round of attribute revision. If either fires, the
finding is fed back into another blocking round (as `prior_linking_findings`, via
`run_blocking_agent`'s `linking_feedback` parameter) and the whole pipeline runs again —
bounded by `AGENT_MAX_OUTER_ROUNDS` (default 2: "give re-blocking one chance," not an
open-ended search). Each round is logged to
`data/artifacts/<block>/<run_id>/outer_loop_round<N>.json`, in the same one-run-one-
directory `<run_id>` blocking/attributes/linking's own artifacts land in (see above) —
one `run_outer_loop` call, one shared run directory across every stage it includes.

`--steps` runs only a subset of the three stages against a block already partway
through the pipeline, e.g.:

```bash
# Redo attribute selection (and relink) without touching the existing blocking rule:
uv run scripts/09_run_outer_loop.py --block beans --steps attributes,linking

# Just relink against whatever attributes are already persisted:
uv run scripts/09_run_outer_loop.py --block beans --steps linking
```

A skipped stage isn't rerun with cached results — it's just not touched, and whichever
later stages you did include use its last persisted output as-is. Its diagnostic
artifacts are still copied forward from the most recent previous run into this run's
directory, though, so a `--steps linking` run's folder still looks like a complete
record (blocking/attributes artifacts included, just carried over rather than freshly
produced) instead of one silently missing files every other run has. If no previous
run has those artifacts to copy — the very first run for a block, or the previous run
also skipped that stage — this fails loudly (`FileNotFoundError`) rather than
silently leaving the new run's folder incomplete: run with the stage included at least
once first. The re-blocking feedback loop above only fires when `--steps` includes both
`blocking` and `linking` (nothing to re-block *in response to* otherwise) — with either
excluded, it runs a single round and logs a warning instead of looping if a
blocking-shaped problem is found but can't be acted on this run.

## Visualizing matches (`scripts/08_visualize_matches.py`)

Wraps splink's own Altair/HTML chart methods (`linking/charts.py`) for a block's
*current* attribute set (`attributes/generated/<block>/latest.json`), retraining fresh
each call since nothing in this pipeline persists a trained model to disk (same as
`linking/evaluate.py`). Output goes to `data/artifacts/chart_<kind>_<block>.html` —
open directly in a browser.

- `waterfall --block yogurt [--n 10] [--mode stratified|top|bottom|borderline] [--threshold 0.0]`
  — how each comparison contributed to the final match score, for a selection of pairs.
  `stratified` (default) spreads the selection evenly across the whole score range so
  the chart shows a representative mix of clear matches, clear non-matches, and
  borderline cases, rather than N near-identical high-confidence pairs.
- `weights --block yogurt` — the model's learned strength-of-evidence per comparison
  level (doesn't need predictions, just the trained model).
- `histogram --block yogurt [--threshold 0.0]` — distribution of match weights across
  every predicted pair in the block.
- `dashboard --block yogurt [--num-example-rows 3]` — splink's interactive
  comparison-viewer dashboard; the most thorough option for manual SME review, at the
  cost of a larger HTML file (a few MB).

The waterfall chart specifically requires splink's
`retain_intermediate_calculation_columns=True`, which the main training path
(`linking/splink_model.py::build_linker`) otherwise defaults to `False` for (see
"Known constraints" below) — `charts.py` passes it explicitly for its own,
separately-trained linker instance. This is safe at this project's bounded block sizes
(verified: ~3.3GB peak for yogurt's ~660K candidate pairs) since the actual OOM cause
was the *unbounded EM blocking* fixed in `train()`, not this flag by itself.

## Corpus profiling: grounding LLM proposals in real catalog statistics

`profiling.py` computes, once (as part of `scripts/03_build_fdc_db.py`), exact
token document-frequency and categorical field distributions over the **full** FNDDS
and OFF datasets (fast even at OFF's ~4.66M rows — DuckDB's vectorized execution does
this in well under a second, so no sampling is needed), and persists them to
`data/profiling/`.

Without this, the blocking and matching-attribute agent loops only ever showed the LLM
(real or mock) a few dozen sample records per round — enough to mine plausible-looking
keywords/categories from, but with no way to tell "specific to this block" apart from
"common across the whole catalog". A keyword like `protein`, `rice`, or `black` looks
harmless in a 40-row yogurt/beans sample but independently matches 20K-65K unrelated OFF
records catalog-wide. Two things are now grounded in the precomputed stats instead of
guesswork:

- **Blocking** (`blocking/agent_loop.py`): each round's prompt includes `corpus_stats`
  — each side's catalog size plus its catalog-wide-common terms, with the system prompt
  instructing the LLM to avoid proposing any of them as a standalone keyword unless the
  block's own name. For the OFF side specifically, `llm/mock.py` also *enforces* this as
  a hard rejection (`profiling.OFF_GENERIC_TERM_MIN_DOC_COUNT` = 15,000 estimated
  matches) rather than just suggesting it, since the mock can't reason about breadth the
  way a real LLM should. There's deliberately no equivalent hard rejection for FNDDS —
  it's only ~5.4K rows total, so even a broad FNDDS keyword isn't a memory risk, and
  applying the same bar there only throws away good keywords (e.g. "cooked", "canned")
  for no safety benefit.
- **Matching attributes** (`attributes/agent_loop.py`): each round's prompt includes
  `field_stats` — the most common real values of each side's categorical fields
  (OFF `categories_tags`/`brands`, FNDDS's WWEIA category) *within this block's
  population*, so a proposed categorical attribute (e.g. `bean_type`'s categories) can
  be grounded in values that actually occur rather than invented from world knowledge.

`data/profiling/` is derived data (gitignored, like `data/blocks/` etc.) — rebuild with
`uv run scripts/03_build_fdc_db.py` or `uv run python -m agentic_matching.profiling`.

## Structured category-based blocking (not just free-text keywords)

Free-text keyword matching against FNDDS's blob of description + WWEIA category +
"Additional Description" text turned out to be badly exploitable: FNDDS's
`additional_description` field is full of boilerplate variant-annotations ("all
flavors", "multigrain, whole grain, whole wheat") shared across many unrelated food
categories, and even a keyword mined from clean description text (e.g. "fruit",
"whole", "plain") independently recurs in countless other foods' own descriptions too
(fruit salad, whole wheat muffins, plain pretzels). Concretely, on this project's own
data: the yogurt block's FNDDS side was **1,176 records, of which 1,114 (95%) were not
yogurt at all** — chicken, coffee, pasta, tea, sandwiches — all pulled in by "flavors"/
"fruit"/"plain"/"whole" matching boilerplate annotation text having nothing to do with
the food's actual identity. OFF had the same failure mode from a single mined keyword
("almond" alone pulled in "Milk Chocolate With Caramelized Almonds").

Both datasets actually carry clean, human-curated categorical labels for exactly this
kind of thing — FNDDS's WWEIA food category (e.g. "Yogurt, regular", "Yogurt, Greek")
and OFF's `categories_tags` (e.g. `en:yogurts`) — so a blocking rule can now specify
`"categories"` per side (`blocking/rules.py`), OR'd with the keyword predicate:
FNDDS matches by exact `wweia_food_category_description` equality, OFF by
`categories_tags` array containment. `blocking/agent_loop.py::_category_options` surfaces
the real category values seen among records already plausibly in the block (same
seed-term-filtered population used for keyword-mining samples) as `category_options` in
the prompt, so the LLM (or mock) picks from real values rather than inventing category
names. The FNDDS-side keyword match is also now scoped to the raw `description` column
only (not the boilerplate-laden blob) as a second, complementary fix.

`llm/mock.py` goes one step further: when a clean matching category is found for a
side, it **skips speculative keyword mining entirely** for that side and only proposes
the seed vocabulary + the category — this mock has no way to tell "whole" (bad, matches
unrelated foods) apart from "greek" (good, block-specific) the way a real LLM's world
knowledge could, so when a reliable structured signal exists, trusting it beats
guessing. Net result on this project's data: yogurt's FNDDS side went from 1,176
records (95% wrong) to **61 records, 0 wrong**; beans went from 908 to 188, and the 26
"non-bean-worded" survivors are legitimate legume-family foods (chickpeas, lentils,
split peas) correctly captured via the "Beans, peas, legumes" WWEIA category — genuine
recall, not noise. A real LLM should be able to keep mining keywords selectively
alongside categories (recognizing which mined words are block-specific vs. generic)
rather than switching mining off entirely; the mock's all-or-nothing rule is a
known simplification.

Note the calibration-proxy metrics (pair completeness) can't fully reflect this: Branded
Foods (the FNDDS-side text stand-in for calibration, see below) has no WWEIA-equivalent
categorization, so the FNDDS-side category predicate can't be evaluated against that
proxy (`blocking/metrics.py::pair_completeness` passes `category_col=None` for FNDDS
accordingly). The precision win above was confirmed by direct inspection of the
materialized block, not by the automatic metric — exactly the kind of thing the
"plausibility spot-check" review step (see `PLAN.md`) exists to catch.

## Mining candidate matching attributes from the block itself

For a from-scratch block (no `seed_rules.SEED_ATTRIBUTES` entry, e.g. `beans`), the
initial round previously proposed attributes purely from world knowledge (or, for the
mock, a hand-curated list) — no mechanism actually *selected* them from the data.
`attributes/agent_loop.py::_candidate_boolean_terms` mines the block's own free text
(FNDDS `description`, OFF `search_text`) for tokens that split its population into a
meaningful minority/majority **on at least one side** (a `min_frac`-`max_frac` band —
near-0% or near-100% doesn't discriminate anything), ranked by the *minimum* of the two
sides' fractions so terms with corroborating signal on both sides (real cross-dataset
concepts, e.g. "meat") outrank one-sided noise (FNDDS's beans block also catches some
unrelated "mixed dish" categories — pasta, potato, sandwich dishes that happen to
mention beans — whose incidental vocabulary would otherwise dominate a pure-frequency
ranking). Passed into every round's prompt as `candidate_terms`.

`llm/mock.py` now uses this instead of a hand-curated boolean attribute list: for a
from-scratch block, it combines a small hand-curated *categorical* exception
(`_CATEGORICAL_EXCEPTIONS` — currently just `beans`' `bean_type`/`sodium_level`, kept
because grouping domain synonyms like "garbanzo"/"chickpea"/"pois chiche" into one
category is exactly the kind of reasoning frequency-counting can't do) with boolean
attributes built directly from the top mined `candidate_terms` (e.g. `has_meat`,
`has_canned`). Redundant mined attributes still get caught by the existing
`metrics.py` step in a later round exactly as if the LLM had proposed them
(verified: mining once surfaced `has_black`, which is redundant with `bean_type`'s
"black" category — flagged at Cramér's V 0.972 and dropped in round 1, unprompted).

**Limitations honestly worth knowing:** a single mined token only catches literal
occurrences of that word — `has_meat` (mined) only fires on "meat" itself, while a
hand-written `has_meat` attribute could list "pork"/"beef"/"bacon"/"sausage"/"ham" as
synonyms, catching far more records. Mining also can't guarantee every intuitively
useful attribute surfaces — in this project's actual data, `rice` mined at rank ~16
(just outside the mock's top-6 cutoff) because its OFF-side signal in this block
happened to be weaker (~1.2%) than `meat`'s (~5.3%), so `with_rice` doesn't currently
appear via mining alone. **This is exactly the gap a real LLM should close** (see below)
— it isn't limited to literal substring co-occurrence and can reason "rice matters for
a beans-and-rice mixed dish" the way it could reason "porc" means "pork".

### What changes with a real LLM instead of the mock?

**Nothing else in the codebase.** `corpus_stats`, `field_stats`, and `candidate_terms`
are assembled by `blocking/agent_loop.py`/`attributes/agent_loop.py` and handed to
`ChatClient.complete_json` as plain prompt text (see `llm/prompts.py`) — the same
payload reaches `llm/mock.py` and a real Ollama-backed `llm/client.py` model alike.
Switching between them is the one-variable change described below; no prompt,
agent-loop, correlation-check, or splink code needs to change. What *does* change is
quality: a real LLM sees the same `candidate_terms` grounding but can go beyond literal
substring frequency — proposing `with_rice` even though "rice" mined at a middling rank,
recognizing "pork"/"beef"/"bacon" belong under `has_meat`, and recognizing
cross-language synonyms ("porc", "riz") the mock's mining explicitly disclaims it can't
do.

## LLM backend

Every agent-loop script calls `get_llm_client()`, selected by `LLM_DEVICE`.

- **`LLM_DEVICE=ollama` (default)**: talks to a local or remote
  [Ollama](https://ollama.com/download) server (a single installer, no manual build
  step) over its OpenAI-compatible `/v1` API. `uv run python -m agentic_matching.llm.server`
  starts (or attaches to) it in the foreground — run it in its own terminal before
  `scripts/05-09`, which only ever *talk* to a server, never start one themselves.
  `llm/server.py::OllamaServerManager` handles two things worth knowing: (1) Ollama is
  commonly already running as a persistent background service (the official installer
  sets up a systemd service on Linux) — it detects an already-reachable server and
  reuses it instead of erroring or double-launching, only spawning its own `ollama
  serve` if nothing answers yet, and only stops a server it started itself; (2) Ollama
  needs a model *pulled* before it can serve it, so `start()` runs `ollama pull <model>`
  automatically (a fast no-op if already present). Defaults to `LLM_MODEL=qwen2.5:1.5b`
  and `LLM_PORT=11434` (Ollama's own conventions) unless you override them in `.env`.
  Requires a reasonably recent Ollama version for its OpenAI-compatible `/v1` API
  (including `response_format` JSON-mode support, which `llm/client.py` relies on) --
  verified against a real Ollama install over the course of this project's development.
- **`LLM_DEVICE=databricks`**: a Databricks Model Serving pay-per-token endpoint.
  Credentials come from the same `DATABRICKS_HOST`/`DATABRICKS_TOKEN` env var names the
  Databricks CLI/SDK themselves use (not duplicated under an `LLM_`-prefixed name);
  `DATABRICKS_LLM_ENDPOINT` is either a bare serving-endpoint name or the full
  invocations URL copy-pasted from the Databricks UI's "Query endpoint" page — see
  `.env.example`. Nothing to start/manage — it's already running as a cloud service.
  Every call is a direct HTTP POST (not the `openai` SDK's automatic URL construction)
  to that endpoint's own literal `.../serving-endpoints/<name>/invocations` URL — see
  `llm/client.py`'s module docstring for why: Databricks' documented shared-gateway
  pattern (routing by a `model` field in the request body) returned an HTML login-page
  redirect for a real named endpoint on a real workspace, while the literal per-endpoint
  URL returned a real API response; the token also needs `model-serving`/
  `model-serving-inference` scope, which a scoped OAuth/service-principal token may not
  have by default even though a full personal access token typically does. Verified
  end-to-end against a real endpoint (`databricks-meta-llama-3-3-70b-instruct`).
  **This is a paid endpoint** — every LLM call in every agent loop (many per block run)
  costs real money while this is set; switch back to `LLM_DEVICE=ollama` when you're
  not deliberately using it.
- `LLM_DEVICE=mock`: offline, no server required. `llm/mock.py` implements the same
  `ChatClient` interface with deterministic keyword-mining heuristics, enough to
  exercise/test/demo the full pipeline end-to-end without any LLM installed.
- Set `LLM_BASE_URL` instead to point at an already-running OpenAI-compatible server
  (e.g. a remote host, or a separately-managed Ollama instance) rather than launching
  one locally -- doesn't apply to `LLM_DEVICE=databricks`, which always uses its own
  literal invocations URL regardless (see above).

## Known constraints at this project's data scale

The pipeline pre-filters to per-block subsets (`data/blocks/<block>_{fndds,off}.parquet`,
written once by `scripts/05_run_blocking_agent.py`) before splink ever runs, so splink
never touches the full ~4.66M-row OFF table directly — but the blocks are still
asymmetric (hundreds to low thousands of FNDDS records vs. tens of thousands of OFF
records), and a few things below turned out to matter at that scale. All are fixed, but
worth knowing if this pipeline is extended (larger blocks, more attributes, etc.), and
worth double-checking on a memory-constrained machine in particular:

- **EM blocking on a skewed attribute alone.** `linking/splink_model.py`'s EM passes
  deliberately combine any attribute column with the search-text prefix condition
  (`block_on(col, "substr(l.search_text, 1, 4)")`) rather than blocking on the attribute
  alone — blocking on a skewed boolean attribute by itself (e.g. `is_greek`, False for
  the vast majority of records) pairs up tens of millions of records and was what
  originally exhausted memory during development.
- **Uncapped exhaustive holdout scoring.** `linking/evaluate.py`'s holdout scoring uses
  an exhaustive (`1=1`) blocking rule, safe only because the holdout sample size is
  capped (`max_holdout_positives`, default 500) — Branded Foods republishes many rows
  per GTIN, so an uncapped category holdout can be tens of thousands of rows, and an
  exhaustive cross join at that scale is the other thing that exhausted memory during
  development.
- **Nondeterministic keyword-mining samples.** `blocking/agent_loop.py::_sample_texts`
  now `ORDER BY fdc_id`/`code` explicitly. Without it, `LIMIT` alone left row order (and
  therefore which records get sampled for keyword mining, mock or real LLM) up to
  DuckDB's query plan — observed to swing a block's OFF-side size by 2-3x across runs
  of identical code.
- **Overly-broad mined keywords.** A plain stopword list doesn't generalize here (new
  generic words like `protein`, `rice`, `black`, `green` kept slipping through and each
  independently matched 20K-65K OFF records), and a plain *relative* frequency threshold
  is too loose at OFF's ~4.66M-row scale (even a "rare" token implies a large absolute
  match count) — see the "Corpus profiling" section above for the actual fix
  (`profiling.py` + `corpus_stats` in the prompt + a hard rejection in `llm/mock.py` for
  the OFF side). This only *enforces* on the mock; a real LLM should reason about
  keyword breadth on its own given the same `corpus_stats`, but if a proposed rule still
  produces an outsized block, check `data/blocks/<block>_off.parquet`'s row count before
  running the linking stage.

If `scripts/07_...` still runs long or memory climbs unbounded, stop it and check
`data/blocks/<block>_off.parquet` row counts — a much larger OFF-side block than the
ones checked in here may need tighter blocking before EM.

- **TODO: linking holdout exclusion is seed-rule-only, not rule-actual.**
  `linking/evaluate.py::_load_block_holdout` filters out known false-positive
  categories from the calibration holdout using `blocking/seed_rules.json`'s
  `exclude_keywords` — but a block with no seed entry (or a seed whose excludes
  haven't caught up with what the block's real, currently-materialized rule has
  since learned) silently gets less exclusion applied than that rule would give it.
  Verified real case: `beans` had no seed entry at all for most of this project's
  history, and 162 real jelly-bean-candy rows (`branded_food_category=Candy`, OFF tag
  `en:jelly-beans`) polluted its holdout f1/attribute_discriminative_power/
  holdout_error_examples calculations as a result — fixed for now by adding/
  extending `beans`' seed entry with the excludes its own round-1 blocking rule had
  already independently learned (`jelly`, `vanilla bean`, `protein powder`,
  `crisps`), but this is a per-block patch, not a systemic fix: any future block
  without a hand-curated seed (or whose seed drifts stale relative to its actual
  rule) will hit the same silent gap. A more systemic fix — drawing holdout excludes
  from the block's actual current/best rule instead of (or in addition to) the seed
  — was deliberately deferred: that rule is itself LLM-proposed, so using it to
  define the ground truth the same LLM's attribute revisions get scored against
  risks a milder version of the exact circularity `blocking/metrics.py::
  term_predicate_sql`'s docstring already avoids on the inclusion side (an overly
  aggressive learned exclude list could shrink the holdout toward the cases the
  current attributes already handle well, quietly inflating holdout f1 with the
  model's own choices). It also needs a persisted canonical "final rule" artifact
  for blocking, which doesn't exist today (unlike attributes'
  `generated/<block>/latest.json`) — `blocking/agent_loop.py` picks a final rule
  in-memory and never writes it out separately from the per-round artifacts.

## Testing

```bash
uv run pytest
```
