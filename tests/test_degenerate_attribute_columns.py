from agentic_matching.linking.degeneracy_check import degenerate_attribute_columns


def test_no_flags_returns_empty():
    assert degenerate_attribute_columns([], {"is_greek"}) == []


def test_irrelevant_flag_kind_ignored():
    flags = [{"kind": "degenerate_prior", "column": "is_greek"}]
    assert degenerate_attribute_columns(flags, {"is_greek"}) == []


def test_collapsed_description_not_an_attribute_name_excluded():
    flags = [{"kind": "collapsed", "column": "description"}]
    assert degenerate_attribute_columns(flags, {"is_greek", "fat_level"}) == []


def test_collapsed_attribute_column_returned():
    flags = [{"kind": "collapsed", "column": "beans_is_bean_dip"}]
    assert degenerate_attribute_columns(flags, {"beans_is_bean_dip", "fat_level"}) == ["beans_is_bean_dip"]


def test_label_switching_attribute_column_returned():
    # Verified real case: beans_preparation_method showed label_switching in every
    # round of the block's history (true pairs agreed 0% of the time vs 0.2% for
    # random decoys) -- see degenerate_attribute_columns's docstring.
    flags = [{"kind": "label_switching", "column": "beans_preparation_method"}]
    assert degenerate_attribute_columns(flags, {"beans_preparation_method"}) == ["beans_preparation_method"]


def test_untrained_attribute_column_returned():
    flags = [{"kind": "untrained", "column": "fat_level"}]
    assert degenerate_attribute_columns(flags, {"fat_level"}) == ["fat_level"]


def test_label_switching_on_description_excluded_by_default():
    # description's 3-level JaroWinklerAtThresholds comparison is structurally
    # different from an attribute's 2-level ExactMatch -- label_switching there is a
    # blocking-shaped question, not "this attribute has no signal", so it must never
    # be returned regardless of `kinds`.
    flags = [{"kind": "label_switching", "column": "description"}]
    assert degenerate_attribute_columns(flags, {"fat_level"}) == []


def test_mixed_flags_only_attribute_ones_returned():
    flags = [
        {"kind": "collapsed", "column": "description"},
        {"kind": "collapsed", "column": "beans_is_bean_dip"},
        {"kind": "label_switching", "column": "beans_preparation_method"},
        {"kind": "untrained", "column": "some_other_attr"},
    ]
    result = degenerate_attribute_columns(flags, {"beans_is_bean_dip", "beans_preparation_method", "some_other_attr"})
    assert result == ["beans_is_bean_dip", "beans_preparation_method", "some_other_attr"]


def test_custom_kinds_restricts_to_collapsed_only():
    # outer_loop.diagnose_blocking_problem's use case: only "collapsed" should count,
    # matching its own narrower, already-documented scope.
    flags = [
        {"kind": "collapsed", "column": "beans_is_bean_dip"},
        {"kind": "label_switching", "column": "beans_preparation_method"},
    ]
    result = degenerate_attribute_columns(
        flags, {"beans_is_bean_dip", "beans_preparation_method"}, kinds=frozenset({"collapsed"})
    )
    assert result == ["beans_is_bean_dip"]
