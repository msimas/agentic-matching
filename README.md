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
uv run scripts/03_build_fdc_db.py          # data/fdc.duckdb: unified per-dataset views, + profiling
uv run scripts/04_build_calibration.py     # Branded<->OFF gold pairs, train/holdout split
uv run scripts/05_run_blocking_agent.py --block yogurt   # (or beans)
uv run scripts/06_run_matching_agent.py --block yogurt
uv run scripts/07_run_splink_and_evaluate.py --block yogurt
uv run scripts/08_visualize_matches.py waterfall --block yogurt   # or: weights | histogram | dashboard
```

Each agent-loop script logs every round to `data/artifacts/` for SME review:
`blocking_<block>_round<N>.json` and `attributes_<block>_round<N>.json` hold the
proposed rule/attributes plus their metrics, and `linking_<block>_round<N>.json` holds
degeneracy flags, holdout evaluation, and a plausibility summary (score distribution,
top/bottom-10 examples). For the linking stage specifically, `scripts/07_...` also
writes every predicted FNDDS<->OFF pair (not just the JSON's top/bottom-10) to
**`data/artifacts/matches_<block>_round<N>.csv`** — sorted by `match_probability`
descending, with each matching attribute's value on both sides alongside it — open
that directly in a spreadsheet to inspect the actual match results.

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
- **Matching attributes** (`attributes/generator.py`): each round's prompt includes
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

For a from-scratch block (no `library.SEED_ATTRIBUTES` entry, e.g. `beans`), the
initial round previously proposed attributes purely from world knowledge (or, for the
mock, a hand-curated list) — no mechanism actually *selected* them from the data.
`attributes/generator.py::_candidate_boolean_terms` mines the block's own free text
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
`correlation_check.py` step in a later round exactly as if the LLM had proposed them
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
are assembled by `blocking/agent_loop.py`/`attributes/generator.py` and handed to
`ChatClient.complete_json` as plain prompt text (see `llm/prompts.py`) — the same
payload reaches `llm/mock.py` and a real `llm/client.py`-backed model over
`LLM_DEVICE=cpu`/`cuda`/`rocm`. Switching backends is the one-variable change described
below; no prompt, agent-loop, correlation-check, or splink code needs to change. What
*does* change is quality: a real LLM sees the same `candidate_terms` grounding but can
go beyond literal substring frequency — proposing `with_rice` even though "rice" mined
at a middling rank, recognizing "pork"/"beef"/"bacon" belong under `has_meat`, and
recognizing cross-language synonyms ("porc", "riz") the mock's mining explicitly
disclaims it can't do.

## LLM backend

Every agent-loop script calls `get_llm_client()`, selected by `LLM_DEVICE`. Whichever
backend, `uv run python -m agentic_matching.llm.server` starts (or attaches to) the
server in the foreground — run it in its own terminal before `scripts/05-07`, which
only ever *talk* to a server, never start one themselves.

- `LLM_DEVICE=cpu` (default): launches a local `vllm serve` subprocess. **`vllm` is not
  a project dependency** — its CPU build isn't a normal PyPI wheel (the default `vllm`
  wheel bundles CUDA and expects an NVIDIA GPU) and must be installed separately
  following vLLM's CPU-backend instructions, which can be finicky to get working (build
  from source, specific `VLLM_TARGET_DEVICE=cpu` flags, etc. — see vLLM's own docs). An
  8B model's CPU inference is also slow and memory-heavy; on a resource-constrained box,
  prefer `LLM_DEVICE=mock` for development and only switch to a real backend when you
  actually need live LLM reasoning.
- **`LLM_DEVICE=ollama`**: an easier path to a working CPU backend if vLLM's CPU build
  gives you trouble — [Ollama](https://ollama.com/download) is a single installer, no
  manual build step. `llm/server.py::OllamaServerManager` handles two things vLLM
  doesn't need to: (1) Ollama is commonly already running as a persistent background
  service (the official installer sets up a systemd service on Linux) — it detects an
  already-reachable server and reuses it instead of erroring or double-launching, only
  spawning its own `ollama serve` if nothing answers yet, and only stops a server it
  started itself; (2) Ollama needs a model *pulled* before it can serve it (unlike
  `vllm serve <model>`, which downloads on demand) — it runs `ollama pull <model>`
  automatically (a fast no-op if already present). Defaults to `LLM_MODEL=qwen2.5:1.5b`
  and `LLM_PORT=11434` (Ollama's own conventions) unless you override them — see
  `.env.example`'s comments on leaving these unset vs. pinned so the right default
  applies. **Not exercised against a real Ollama install in this repo's own
  testing** (Ollama isn't installed in this dev environment) — it requires a reasonably
  recent Ollama version for its OpenAI-compatible `/v1` API (including `response_format`
  JSON-mode support, which `llm/client.py` relies on); if something doesn't line up,
  check your installed version's docs.
- `LLM_DEVICE=cuda` / `rocm`: vLLM again, GPU launch flags — a one-variable switch, see
  `src/agentic_matching/config.py`.
- `LLM_DEVICE=mock`: offline, no server required. `llm/mock.py` implements the same
  `ChatClient` interface with deterministic keyword-mining heuristics, enough to
  exercise/test/demo the full pipeline end-to-end without any LLM installed. This is
  what `scripts/05-07` were run with in this repo's checked-in `data/artifacts/`.
- Set `LLM_BASE_URL` instead to point at an already-running OpenAI-compatible server
  (e.g. on a remote GPU box, or a separately-managed Ollama/vLLM) rather than launching
  one locally — works the same way for every `LLM_DEVICE` value.

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

## Testing

```bash
uv run pytest
```
