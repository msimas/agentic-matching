import pandas as pd

from agentic_matching.attributes.generator import _candidate_boolean_terms

# Synthetic block: 10 FNDDS rows, 10 OFF rows. "meat" appears in a meaningful minority
# on both sides; "beans" (the block's own canonical term) appears in nearly all rows on
# both sides; "with" is a stopword; "rare" appears in only one row (below the band).
FNDDS_DESCRIPTIONS = [
    "Beans, black, with meat",
    "Beans, black, with meat",
    "Beans, black, canned",
    "Beans, pinto, canned",
    "Beans, pinto, dried",
    "Beans, kidney, dried",
    "Beans, kidney, plain",
    "Beans, navy, plain",
    "Beans, navy, plain",
    "Beans, rare, plain",
]
OFF_SEARCH_TEXT = [
    "beans with meat sauce",
    "beans with meat sauce",
    "beans canned black",
    "beans canned pinto",
    "beans dried pinto",
    "beans dried kidney",
    "beans plain kidney",
    "beans plain navy",
    "beans plain navy",
    "beans plain navy",
]


def _dfs():
    return (
        pd.DataFrame({"description": FNDDS_DESCRIPTIONS}),
        pd.DataFrame({"search_text": OFF_SEARCH_TEXT}),
    )


def test_finds_meaningful_split_term():
    fndds_df, off_df = _dfs()
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans")
    names = {t["term"] for t in terms}
    assert "meat" in names


def test_excludes_canonical_block_term():
    fndds_df, off_df = _dfs()
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans")
    names = {t["term"] for t in terms}
    assert "beans" not in names
    assert "bean" not in names


def test_excludes_near_universal_term():
    # "plain" appears in 5/10 fndds rows -- within band -- but let's check a term at
    # the edges: nothing here is >90%, so just confirm canonical exclusion holds even
    # though "beans" is literally 100% on both sides (would otherwise pass the band).
    fndds_df, off_df = _dfs()
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans", max_frac=0.99)
    names = {t["term"] for t in terms}
    assert "beans" not in names


def test_excludes_rare_below_band():
    fndds_df, off_df = _dfs()
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans", min_frac=0.15)
    names = {t["term"] for t in terms}
    assert "rare" not in names  # only 1/10 fndds rows, 0/10 off rows


def test_ranks_mutual_signal_above_one_sided_noise():
    # "meat" appears on both sides (mutual signal); construct a one-sided-only term
    # with a higher combined frequency to confirm it doesn't outrank "meat".
    fndds = FNDDS_DESCRIPTIONS + ["Beans, sandwich, onesided"] * 4
    off = OFF_SEARCH_TEXT + ["beans plain navy"] * 4  # no "onesided" on the off side
    fndds_df = pd.DataFrame({"description": fndds})
    off_df = pd.DataFrame({"search_text": off})
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans")
    order = [t["term"] for t in terms]
    assert order.index("meat") < order.index("onesided")


def test_result_capped_at_k():
    fndds_df, off_df = _dfs()
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans", k=2)
    assert len(terms) <= 2


def test_handles_missing_values():
    fndds_df = pd.DataFrame({"description": FNDDS_DESCRIPTIONS + [None, float("nan")]})
    off_df = pd.DataFrame({"search_text": OFF_SEARCH_TEXT + [None, float("nan")]})
    terms = _candidate_boolean_terms(fndds_df, off_df, "beans")
    assert isinstance(terms, list)
