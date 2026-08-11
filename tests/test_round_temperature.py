from agentic_matching.config import llm_settings, round_temperature


def test_round_zero_with_no_prior_state_uses_exploration_temperature():
    assert round_temperature(0, has_prior_state=False) == llm_settings.exploration_temperature


def test_round_zero_with_prior_state_uses_default_temperature():
    # A seeded round 0 (e.g. yogurt's SEED_ATTRIBUTES, or a blocking seed_rules.json
    # entry) is a revision of existing domain knowledge, not a from-scratch proposal.
    assert round_temperature(0, has_prior_state=True) == llm_settings.temperature


def test_later_round_always_uses_default_temperature_regardless_of_prior_state():
    assert round_temperature(1, has_prior_state=False) == llm_settings.temperature
    assert round_temperature(3, has_prior_state=True) == llm_settings.temperature


def test_exploration_temperature_is_higher_than_default():
    # The whole point: more sampling diversity for a genuine blank-slate proposal.
    assert llm_settings.exploration_temperature > llm_settings.temperature
