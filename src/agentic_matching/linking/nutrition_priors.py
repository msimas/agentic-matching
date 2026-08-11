"""Domain-knowledge priors for attributes that track a nutritionally-significant
ingredient -- the addition (or absence) of meat/dairy/egg (nutrient-dense) or
sugar/molasses/honey/syrup (calorie-dense) meaningfully changes a product's nutrition
profile, so two products differing only on one of these should be treated as more
likely to be genuinely different products than EM's raw estimate alone might capture
from a small/unlucky calibration sample -- and conversely, two products agreeing on
one of these terms are less likely to coincidentally agree by chance than a generic
attribute would be.

Unlike a hard per-attribute override (fix_m_probability/fix_u_probability, bypassing
EM training entirely for that comparison), this blends EM's data-driven estimate with
a fixed domain-prior target, weighted by how much calibration data actually backs the
EM estimate (attribute_discriminative_power's n_true_pairs) -- more data lets EM's own
finding dominate more, but the prior always retains some influence (MAX_PRIOR_WEIGHT <
1.0) even at a well-populated sample, since these ingredient categories are asserted
to always carry real nutrition-significance, not just when data happens to be sparse.

Applied as a pure post-processing step on the already-exported trained_settings dict
(same shape degeneracy_check.py/evaluate.py already operate on) -- EM trains
completely unmodified first, so every comparison (including nutrition-significant
ones) gets a real, data-informed estimate; only afterward are the flagged comparisons'
levels nudged toward the prior before the final prediction Linker is built from the
adjusted settings (see splink_model.linker_from_settings).

Detection is name/description substring matching, not a per-block/per-attribute
registry -- see classify_nutrition_significance's docstring for why, and for the
tradeoff against also scanning keyword lists.
"""

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
    }
)
CALORIE_DENSE_TERMS = frozenset({"sugar", "molasses", "honey", "syrup", "corn syrup", "brown sugar"})

# (m_probability, u_probability) prior targets per group. Nutrient-dense ingredients
# (meat/dairy/egg) are treated as a more categorical, near-binary difference (a
# product either has real animal-protein/dairy content or it doesn't) than
# calorie-dense ones (sugar/molasses content is more often a matter of degree --
# "lightly sweetened" vs "unsweetened" isn't as clear-cut as "has bacon" vs "doesn't")
# -- so the calorie-dense prior is deliberately weaker (closer to a generic
# attribute's typical behavior) than the nutrient-dense one.
NUTRIENT_DENSE_PRIOR = (0.93, 0.15)
CALORIE_DENSE_PRIOR = (0.85, 0.25)

# Blend weight (trust in EM's own estimate) scales linearly with observed true-pair
# count up to this cap -- below it, the prior increasingly dominates; the cap itself
# is < 1.0 (not "full trust once past a threshold") because these categories are
# asserted to always carry some real nutrition-significance the calibration sample
# alone shouldn't be allowed to fully override, no matter how much data it has.
PRIOR_CONFIDENCE_PAIRS = 200
MAX_PRIOR_WEIGHT = 0.8


def classify_nutrition_significance(attr: dict[str, Any]) -> str | None:
    """Returns "nutrient_dense", "calorie_dense", or None.

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
    if any(term in text for term in NUTRIENT_DENSE_TERMS):
        return "nutrient_dense"
    if any(term in text for term in CALORIE_DENSE_TERMS):
        return "calorie_dense"
    return None


def _blend(m_data: float, u_data: float, n_true_pairs: int, prior: tuple[float, float]) -> tuple[float, float]:
    prior_m, prior_u = prior
    weight = min(MAX_PRIOR_WEIGHT, n_true_pairs / PRIOR_CONFIDENCE_PAIRS) if PRIOR_CONFIDENCE_PAIRS else 0.0
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
        prior = NUTRIENT_DENSE_PRIOR if group == "nutrient_dense" else CALORIE_DENSE_PRIOR
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
        n_true = n_true_by_attr.get(name, 0)
        m, u = _blend(exact_level["m_probability"], exact_level["u_probability"], n_true, prior)
        log.info(
            "block attribute '%s' (%s): blending EM estimate (m=%.3f u=%.3f, n_true_pairs=%d) "
            "toward nutrition prior (m=%.2f u=%.2f) -> m=%.3f u=%.3f",
            name,
            group,
            exact_level["m_probability"],
            exact_level["u_probability"],
            n_true,
            prior[0],
            prior[1],
            m,
            u,
        )
        exact_level["m_probability"], exact_level["u_probability"] = m, u
        else_level["m_probability"], else_level["u_probability"] = 1 - m, 1 - u
    return settings
