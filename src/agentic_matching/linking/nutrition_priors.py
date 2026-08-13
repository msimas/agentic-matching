"""Domain-knowledge weight adjustments applied on top of EM's own trained m/u, all as
pure post-processing on the already-exported trained_settings dict (same shape
degeneracy_check.py/evaluate.py already operate on) -- EM trains completely
unmodified first, so every comparison gets a real, data-informed estimate; only
afterward are specific comparisons' levels nudged before the final prediction Linker
is built from the adjusted settings (see splink_model.linker_from_settings).

`apply_nutrition_priors` -- attributes that track a nutritionally-significant
ingredient or dietary classification (meat/dairy/egg = nutrient-dense,
sugar/molasses/honey/syrup = calorie-dense, vegetarian/vegan/plant-based = dietary
classification) have their learned m/u blended TOWARD a strong domain prior (an
interpolation, not a hard override like fix_m_probability/fix_u_probability, which
would bypass EM training entirely for a comparison), since this kind of difference
reliably means two products are nutritionally different, more reliably than a
small/unlucky calibration sample may capture.

Detection is name/description substring matching, not a per-block/per-attribute
registry -- see classify_nutrition_significance's docstring for why, and for the
tradeoff against also scanning keyword lists.

This module used to also have `dampen_description_weight`, shrinking `description`'s
own text-similarity comparison toward "no information" because it repeatedly showed
instability (label_switching and collapsed flags) at this project's block scale.
Removed along with `description` itself as a splink comparison (see
splink_model.build_comparisons' docstring for the full verified case) -- damping a
comparison that no longer exists would have been a silent no-op, not a fix."""

from __future__ import annotations

import copy
import logging
from typing import Any

log = logging.getLogger(__name__)

# Not tied to any one block -- applies automatically to any attribute (in any block)
# whose name/description mentions one of these terms, so a future block's LLM-proposed
# "contains_cheese" or "has_honey" attribute is covered without any per-block
# hand-tagging.
NUTRIENT_DENSE_TERMS = frozenset(
    {
        "meat", "beef", "pork", "poultry", "chicken", "turkey", "fish", "bacon",
        "sausage", "dairy", "milk", "cheese", "cream", "butter", "egg", "eggs",
        "fat", "lard", "whey"
    }
)
CALORIE_DENSE_TERMS = frozenset(
    {"sugar", "molasses", "honey", "syrup", "corn syrup", "brown sugar", "fried", "deep-fried", "fat"}
)
# "fried"/"deep-fried"/"fat" briefly lived outside this set (removed entirely) after a
# verified real case: 'breaded_vegetables_vegetable_type' ("The specific vegetable
# being fried or breaded.", a plain categorical identity attribute with nothing to do
# with calorie density) got misclassified as calorie_dense purely because "fried"
# appeared in its description, blending its EM-trained m (~0.9997, a genuinely strong
# attribute) down to ~0.945 toward CALORIE_DENSE_PRIOR -- weak enough, combined with
# this block's low match-probability prior after widening its blocking rule, that NO
# pair ever crossed the 0.5 "predicted match" threshold. Removing the terms fixed that
# one case but was a blunt instrument: it does nothing for the NEXT accidental
# collision (any future block whose defining vocabulary happens to overlap a real
# nutrition term), and it throws away "fried" as a genuine calorie-relevant signal for
# attributes that actually deserve it. Reinstated once `_blend` (below) became aware of
# how strongly the EM-trained comparison ALREADY separates match from non-match --
# see `_blend`'s docstring: an already strongly-separated fit like vegetable_type's
# now gets left almost untouched regardless of classification, so the false-positive
# term match no longer matters, while a genuinely weak/ambiguous fit (the real
# beans_contains_meat case this whole prior mechanism exists for) still gets pulled
# toward the prior as before.
# A product's vegetarian/vegan/plant-based status is arguably the MOST totalizing
# nutrition-relevant distinction of the three groups here -- it isn't just one
# ingredient's presence, it's a claim about the product's entire composition (a vegan
# product cannot legitimately match anything containing meat/dairy/egg at all) -- see
# _GROUPS' ordering below for why this group is checked first.
DIETARY_CLASSIFICATION_TERMS = frozenset({"vegetarian", "vegan", "plant-based", "plant based"})

# (m_probability, u_probability) prior targets per group. Nutrient-dense ingredients
# (meat/dairy/egg) are treated as a more categorical, near-binary difference (a
# product either has real animal-protein/dairy content or it doesn't) than
# calorie-dense ones (sugar/molasses content is more often a matter of degree --
# "lightly sweetened" vs "unsweetened" isn't as clear-cut as "has bacon" vs "doesn't")
# -- so the calorie-dense prior is deliberately weaker (closer to a generic
# attribute's typical behavior) than the nutrient-dense one.
# Was (0.93, 0.15) / (0.85, 0.25); pushed further apart per explicit instruction to
# strengthen the nudge -- verified real effect: beans_contains_meat went from
# EM-only m=0.547 u=0.479 (log2(m/u)~=0.19 bits, ~noise) to m=0.772 u=0.249
# (~1.63 bits), and holdout f1 for the block itself (not just this one attribute)
# rose from 0.0066 to 0.0141 in the same real run once description-damping was added
# alongside it. A further push to (0.995, 0.02)/(0.95, 0.08) with MAX_PRIOR_WEIGHT=
# 0.25 was tried and reverted -- it moved beans_contains_meat further (m=0.879
# u=0.135) but the block's own f1 dropped to 0.0103, worse than this setting's
# 0.0141 -- past some point, overriding EM's own (reasonably informative) fit costs
# more than the stronger assertion is worth. This is that sweet spot, verified
# against real data, not a guess -- see PLAN.md-adjacent verification in this
# module's git history if these numbers ever need revisiting.
NUTRIENT_DENSE_PRIOR = (0.97, 0.06)
CALORIE_DENSE_PRIOR = (0.90, 0.15)
# At least as strong as NUTRIENT_DENSE_PRIOR, for the reason given at
# DIETARY_CLASSIFICATION_TERMS above -- a vegetarian/vegan/plant-based claim covers
# the same ground as the nutrient-dense terms (and more), not a lesser version of it.
DIETARY_CLASSIFICATION_PRIOR = (0.98, 0.04)

# Group name -> (term set, prior target), in classification-priority order: checked
# top to bottom, first match wins. dietary_classification is checked BEFORE
# nutrient_dense on purpose -- an attribute like "beans_is_vegetarian" ("Indicates if
# the product is vegetarian (contains no meat)") would otherwise match on "meat" and
# get misclassified as nutrient_dense instead of the (here, more specific and
# stronger) dietary_classification group. None of the three term sets currently
# overlap in practice, but ordering by specificity is the correct default regardless
# -- keeping this as one ordered list (rather than three separate if-checks) also
# makes adding a fourth group later a one-line change instead of a restructure.
_GROUPS: list[tuple[str, frozenset[str], tuple[float, float]]] = [
    ("dietary_classification", DIETARY_CLASSIFICATION_TERMS, DIETARY_CLASSIFICATION_PRIOR),
    ("nutrient_dense", NUTRIENT_DENSE_TERMS, NUTRIENT_DENSE_PRIOR),
    ("calorie_dense", CALORIE_DENSE_TERMS, CALORIE_DENSE_PRIOR),
]
PRIOR_BY_GROUP: dict[str, tuple[float, float]] = {name: prior for name, _, prior in _GROUPS}

# Sample-size blend weight (trust in EM's own estimate) scales linearly with observed
# true-pair count up to this cap -- below it, the prior increasingly dominates; the cap
# itself is < 1.0 (not "full trust once past a threshold") because these categories are
# asserted to always carry some real nutrition-significance the calibration sample
# alone shouldn't be allowed to fully override BY SAMPLE SIZE ALONE, no matter how much
# data it has. Was 0.8; lowered per explicit instruction for a stronger nudge -- for a
# genuinely weak/ambiguous fit (small OR large true-pair count, doesn't matter), the
# prior now always retains at least 55% influence via this path, e.g. a well-populated
# sample like beans_contains_meat's 500 true pairs. (0.25 was also tried and reverted
# -- see the prior-target comment above for the real-data verification of why.)
#
# This is a FLOOR on trusting the data, not a ceiling on it -- see _blend's
# strength_weight, a second, independent path that can push the effective weight above
# this cap when the EM fit's OWN separation (m_data - u_data) already meets or exceeds
# what the prior itself would assert, regardless of sample size. The two paths measure
# different things (how much data went into the fit vs. how confident the fit already
# is) and the stronger signal wins -- see _blend's docstring for the real case this
# exists for.
PRIOR_CONFIDENCE_PAIRS = 200
MAX_PRIOR_WEIGHT = 0.45


def classify_nutrition_significance(attr: dict[str, Any]) -> str | None:
    """Returns "dietary_classification", "nutrient_dense", "calorie_dense", or None
    -- see _GROUPS for the term sets and why dietary_classification is checked first.

    Checked against the attribute's own `name` + `description` only -- not its
    fndds_keywords/off_keywords (or, for a categorical attribute, each category's
    keyword lists). Name/description is what the attribute is semantically DECLARED
    to represent (and what an SME reviewing artifacts actually reads), so this is
    precise and auditable; scanning keyword lists too would catch more real cases
    (e.g. a categorical attribute whose name doesn't mention meat but has a
    "ham-flavored" category) at the cost of false positives from incidental keywords
    unrelated to what the attribute is actually for. Deliberately the narrower,
    safer option for now -- widen to keyword scanning only if a real missed case
    shows up."""
    text = f"{attr.get('name', '')} {attr.get('description', '')}".lower()
    for group, terms, _prior in _GROUPS:
        if any(term in text for term in terms):
            return group
    return None


def _blend(m_data: float, u_data: float, n_true_pairs: int, prior: tuple[float, float]) -> tuple[float, float]:
    """Combines two independent signals for how much to trust the EM-trained fit over
    the domain prior, and uses whichever trusts the data MORE (a floor on the prior's
    influence, not something the two signals average together):

      - `sample_weight`: how much real calibration data went into this fit (unchanged
        from before this function was made strength-aware -- see PRIOR_CONFIDENCE_PAIRS/
        MAX_PRIOR_WEIGHT's docstring). Capped below 1.0 deliberately: a nutrition-
        significant category is asserted to always carry SOME domain-prior influence,
        no matter how much calibration data backs it.
      - `strength_weight`: how much the fit ALREADY separates match from non-match
        (`m_data - u_data`) relative to how much separation the prior itself asserts
        (`prior_m - prior_u`). Not capped below 1.0 -- if the data's own fit is already
        at least as confidently separated as the prior would assert, there's nothing
        left for the prior to usefully correct, so it's let all the way to zero
        influence rather than artificially held back.

    This is what lets a genuinely nutrition-significant TERM (e.g. "fried") stay in the
    classifier's term list without a real, already-strongly-discriminative comparison
    that happens to match it on an incidental word getting dragged down toward the
    prior anyway. Verified real case (breaded_vegetables, LLM_DEVICE=databricks):
    'breaded_vegetables_vegetable_type' ("The specific vegetable being fried or
    breaded.") matched "fried" and got classified calorie_dense, but its EM fit
    (m=0.9997, u=0.077, gap=0.923) already separates far more confidently than
    CALORIE_DENSE_PRIOR itself asserts (gap=0.90-0.15=0.75) -- strength_weight =
    min(1, 0.923/0.75) = 1.0, so the prior contributes nothing and the fit is returned
    essentially untouched, instead of being pulled down to ~0.945 (the sample-size-only
    weight of 0.45's result) and suppressing every match probability below the 0.5
    threshold. A genuinely weak/ambiguous fit -- the real beans_contains_meat case this
    whole prior mechanism was built for (EM-only m=0.547, u=0.479, gap=0.068, far below
    NUTRIENT_DENSE_PRIOR's own 0.91 gap) -- gets a near-zero strength_weight and falls
    back to the unchanged sample-size floor, so that verified, previously-tuned
    behavior is untouched by this addition."""
    prior_m, prior_u = prior
    sample_weight = min(MAX_PRIOR_WEIGHT, n_true_pairs / PRIOR_CONFIDENCE_PAIRS) if PRIOR_CONFIDENCE_PAIRS else 0.0
    prior_gap = prior_m - prior_u
    data_gap = m_data - u_data
    strength_weight = min(1.0, max(0.0, data_gap / prior_gap)) if prior_gap > 0 else 0.0
    weight = max(sample_weight, strength_weight)
    m = weight * m_data + (1 - weight) * prior_m
    u = weight * u_data + (1 - weight) * prior_u
    return m, u


def apply_nutrition_priors(
    trained_settings: dict[str, Any],
    attrs: list[dict[str, Any]],
    discriminative_power: list[dict[str, Any]],
) -> dict[str, Any]:
    """Returns a NEW settings dict (deep-copied; `trained_settings` itself is never
    mutated) with any nutrition-significant attribute's comparison levels blended
    toward its domain prior -- see this module's docstring for the full rationale.

    Every attribute comparison in this codebase has exactly 3 levels (null / exact
    match / else -- see splink_model.build_comparisons, which always uses
    cl.ExactMatch for every attribute regardless of kind), so this blends the
    exact-match level directly and recomputes "else" as its complement (1 - m, 1 - u)
    rather than blending it independently, keeping each level's m/u summing to 1
    across the comparison the way splink expects. A comparison that doesn't have this
    shape (e.g. `description`'s multi-level JaroWinklerAtThresholds, or some future
    non-ExactMatch attribute comparison) is left untouched -- this function only ever
    adjusts attribute comparisons whose column name matches one of `attrs`, and
    `description` is never in that list.
    """
    n_true_by_attr = {d["attribute"]: d.get("n_true_pairs", 0) for d in discriminative_power}
    attr_by_name = {a["name"]: a for a in attrs}

    settings = copy.deepcopy(trained_settings)
    for comparison in settings.get("comparisons", []):
        name = comparison.get("output_column_name")
        attr = attr_by_name.get(name)
        if attr is None:
            continue
        group = classify_nutrition_significance(attr)
        if group is None:
            continue
        prior = PRIOR_BY_GROUP[group]
        levels = [lvl for lvl in comparison.get("comparison_levels", []) if not lvl.get("is_null_level")]
        if len(levels) != 2:
            log.warning(
                "Skipping nutrition-prior blend for '%s': expected a 2-level "
                "ExactMatch comparison, found %d non-null level(s).",
                name,
                len(levels),
            )
            continue
        exact_level, else_level = levels
        # EM doesn't always fully train every comparison every round (e.g. a column
        # blocked-on this round's EM pass, or too little data in a small/sparse
        # round -- "not yet fully trained... no m values are trained" is a real,
        # valid splink outcome, not an error) -- exact_level["m_probability"] can then
        # be missing or None. Verified real crash (beans, LLM_DEVICE=databricks):
        # beans_contains_meat went untrained one round and this raised KeyError,
        # taking down the whole round instead of just skipping the blend for that one
        # attribute. Treat "no EM estimate to blend with" the same as "no data" --
        # trust the prior alone (n_true=0), rather than crash or silently skip the
        # nutrition-significant attribute entirely.
        m_raw, u_raw = exact_level.get("m_probability"), exact_level.get("u_probability")
        untrained = m_raw is None or u_raw is None
        n_true = 0 if untrained else n_true_by_attr.get(name, 0)
        m_data = prior[0] if untrained else m_raw
        u_data = prior[1] if untrained else u_raw
        m, u = _blend(m_data, u_data, n_true, prior)
        log.info(
            "block attribute '%s' (%s): blending EM estimate (m=%s u=%s%s, n_true_pairs=%d) "
            "toward nutrition prior (m=%.2f u=%.2f) -> m=%.3f u=%.3f",
            name,
            group,
            m_raw,
            u_raw,
            " -- UNTRAINED this round, using the prior alone" if untrained else "",
            n_true,
            prior[0],
            prior[1],
            m,
            u,
        )
        exact_level["m_probability"], exact_level["u_probability"] = m, u
        else_level["m_probability"], else_level["u_probability"] = 1 - m, 1 - u
    return settings
