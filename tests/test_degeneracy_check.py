from agentic_matching.linking.degeneracy_check import check_degeneracy


def _settings(probability_two_random_records_match=0.01, levels=None):
    return {
        "probability_two_random_records_match": probability_two_random_records_match,
        "comparisons": [
            {
                "output_column_name": "is_greek",
                "comparison_levels": levels
                or [
                    {"m_probability": 0.9, "u_probability": 0.1},
                    {"m_probability": 0.1, "u_probability": 0.9},
                ],
            }
        ],
    }


def test_healthy_comparison_has_no_flags():
    settings = _settings()
    assert check_degeneracy(settings) == []


def test_detects_label_switching():
    # More-agreement level has LOWER m_probability than the less-agreement level below it.
    settings = _settings(
        levels=[
            {"m_probability": 0.2, "u_probability": 0.1},
            {"m_probability": 0.8, "u_probability": 0.9},
        ]
    )
    flags = check_degeneracy(settings)
    assert any(f["kind"] == "label_switching" for f in flags)


def test_detects_collapsed_comparison():
    settings = _settings(
        levels=[
            {"m_probability": 0.5, "u_probability": 0.5},
            {"m_probability": 0.3, "u_probability": 0.3},
        ]
    )
    flags = check_degeneracy(settings)
    assert any(f["kind"] == "collapsed" for f in flags)


def test_detects_degenerate_prior_near_zero():
    settings = _settings(probability_two_random_records_match=1e-8)
    flags = check_degeneracy(settings)
    assert any(f["kind"] == "degenerate_prior" for f in flags)


def test_detects_degenerate_prior_near_one():
    settings = _settings(probability_two_random_records_match=0.9999)
    flags = check_degeneracy(settings)
    assert any(f["kind"] == "degenerate_prior" for f in flags)


def test_untrained_comparison_flagged():
    settings = _settings(levels=[{"m_probability": None, "u_probability": None}])
    flags = check_degeneracy(settings)
    assert any(f["kind"] == "untrained" for f in flags)
