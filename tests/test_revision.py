from agentic_matching.attributes.revision import (
    ask_with_followup,
    decide_keep_drop,
    define_attributes,
    identify_gap,
    revise_attributes,
)


class QueueClient:
    """Returns each queued response in order, one per complete_json call -- lets a
    test script exactly what the LLM says at each of the three (or more, with a
    follow-up) calls a decomposed revision makes."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def complete_json(self, system, user, *, max_tokens=None, temperature=None):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return self._responses.pop(0)


ATTR_A = {"name": "beans_bean_type", "kind": "categorical", "description": "type", "categories": {"kidney": {"fndds_keywords": ["kidney"], "off_keywords": ["kidney"]}}}
ATTR_B = {"name": "beans_contains_meat", "kind": "boolean", "description": "meat", "fndds_keywords": ["pork"], "off_keywords": ["pork"]}


# -- ask_with_followup -------------------------------------------------------------


def test_ask_with_followup_passthrough_when_no_followup_requested():
    client = QueueClient([{"decisions": []}])
    response = ask_with_followup(client, "sys", "user", 0.1, [], [])
    assert response == {"decisions": []}
    assert len(client.calls) == 1


def test_ask_with_followup_fulfills_and_reasks_once():
    client = QueueClient(
        [
            {"status": "need_more_info", "requested": [{"kind": "term_frequency", "term": "bacon"}]},
            {"decisions": [{"name": "x", "action": "keep"}]},
        ]
    )
    response = ask_with_followup(client, "sys", "user", 0.1, ["bacon beans"], [])
    assert response == {"decisions": [{"name": "x", "action": "keep"}]}
    assert len(client.calls) == 2
    assert "bacon" in client.calls[1]["user"]  # the answer got embedded in the follow-up prompt


def test_ask_with_followup_never_loops_beyond_one_followup():
    # Even if the second response is ALSO a need_more_info, we don't ask a third time.
    client = QueueClient(
        [
            {"status": "need_more_info", "requested": [{"kind": "term_frequency", "term": "x"}]},
            {"status": "need_more_info", "requested": [{"kind": "term_frequency", "term": "y"}]},
        ]
    )
    response = ask_with_followup(client, "sys", "user", 0.1, [], [])
    assert response["status"] == "need_more_info"
    assert len(client.calls) == 2


# -- decide_keep_drop ---------------------------------------------------------------


def test_decide_keep_drop_empty_existing_makes_no_call():
    client = QueueClient([])
    assert decide_keep_drop(client, "beans", [], None, None, None, 0.1, [], []) == {}
    assert client.calls == []


def test_decide_keep_drop_parses_actions():
    client = QueueClient(
        [{"decisions": [{"name": "beans_bean_type", "action": "keep"}, {"name": "beans_contains_meat", "action": "drop"}]}]
    )
    decisions = decide_keep_drop(client, "beans", [ATTR_A, ATTR_B], None, None, None, 0.1, [], [])
    assert decisions == {"beans_bean_type": "keep", "beans_contains_meat": "drop"}


def test_decide_keep_drop_defaults_unmentioned_to_keep():
    client = QueueClient([{"decisions": [{"name": "beans_bean_type", "action": "drop"}]}])
    decisions = decide_keep_drop(client, "beans", [ATTR_A, ATTR_B], None, None, None, 0.1, [], [])
    assert decisions["beans_bean_type"] == "drop"
    assert decisions["beans_contains_meat"] == "keep"  # never mentioned -> conservative default


def test_decide_keep_drop_ignores_invalid_action_value():
    client = QueueClient([{"decisions": [{"name": "beans_bean_type", "action": "delete"}]}])
    decisions = decide_keep_drop(client, "beans", [ATTR_A], None, None, None, 0.1, [], [])
    assert decisions["beans_bean_type"] == "keep"  # invalid action ignored, falls back to default


# -- identify_gap ---------------------------------------------------------------


def test_identify_gap_skipped_without_error_examples_no_llm_call():
    client = QueueClient([])
    assert identify_gap(client, "beans", None, [], None, 0.1, [], []) is None
    assert identify_gap(client, "beans", {"false_positives": [], "false_negatives": []}, [], None, 0.1, [], []) is None
    assert client.calls == []


def test_identify_gap_returns_concept_when_needed():
    client = QueueClient([{"needed": True, "concept": "contains molasses", "rationale": "..."}])
    error_examples = {"false_positives": [{"a": 1}], "false_negatives": []}
    concept = identify_gap(client, "beans", error_examples, [], None, 0.1, [], [])
    assert concept == "contains molasses"


def test_identify_gap_returns_none_when_not_needed():
    client = QueueClient([{"needed": False}])
    error_examples = {"false_positives": [{"a": 1}], "false_negatives": []}
    assert identify_gap(client, "beans", error_examples, [], None, 0.1, [], []) is None


# -- define_attributes ---------------------------------------------------------------


def test_define_attributes_keep_only_makes_no_llm_call():
    client = QueueClient([])
    decisions = {"beans_bean_type": "keep", "beans_contains_meat": "keep"}
    result = define_attributes(client, "beans", [ATTR_A, ATTR_B], decisions, None, [], None, None, None, 0.1)
    assert result == [ATTR_A, ATTR_B]
    assert client.calls == []


def test_define_attributes_drop_omits_attribute():
    client = QueueClient([])
    decisions = {"beans_bean_type": "keep", "beans_contains_meat": "drop"}
    result = define_attributes(client, "beans", [ATTR_A, ATTR_B], decisions, None, [], None, None, None, 0.1)
    assert result == [ATTR_A]


def test_define_attributes_redefine_calls_llm_for_only_that_attribute():
    redefined = {"name": "beans_contains_meat", "kind": "boolean", "description": "meat v2", "fndds_keywords": ["pork", "bacon"], "off_keywords": ["pork", "bacon"]}
    client = QueueClient([{"attributes": [redefined]}])
    decisions = {"beans_bean_type": "keep", "beans_contains_meat": "redefine"}
    result = define_attributes(client, "beans", [ATTR_A, ATTR_B], decisions, None, [], None, None, None, 0.1)
    assert ATTR_A in result
    assert redefined in result
    assert len(client.calls) == 1
    # to_define only mentions the attribute actually being redefined.
    assert "beans_contains_meat" in client.calls[0]["user"]


def test_define_attributes_new_from_gap_concept():
    new_attr = {"name": "beans_has_molasses", "kind": "boolean", "description": "molasses", "fndds_keywords": ["molasses"], "off_keywords": ["molasses"]}
    client = QueueClient([{"attributes": [new_attr]}])
    decisions = {"beans_bean_type": "keep"}
    result = define_attributes(client, "beans", [ATTR_A], decisions, "contains molasses", [], None, None, None, 0.1)
    assert ATTR_A in result
    assert new_attr in result


def test_define_attributes_malformed_definition_filtered_out():
    malformed = {"name": "beans_contains_meat", "kind": "boolean", "description": "broken", "fndds_keywords": [], "off_keywords": []}
    client = QueueClient([{"attributes": [malformed]}])
    decisions = {"beans_contains_meat": "redefine"}
    result = define_attributes(client, "beans", [ATTR_B], decisions, None, [], None, None, None, 0.1)
    assert result == []  # empty keyword lists -> filter_valid_attributes drops it


# -- revise_attributes (full orchestration) ------------------------------------------


def test_revise_attributes_full_flow_keep_and_new():
    new_attr = {"name": "beans_has_molasses", "kind": "boolean", "description": "molasses", "fndds_keywords": ["molasses"], "off_keywords": ["molasses"]}
    client = QueueClient(
        [
            {"decisions": [{"name": "beans_bean_type", "action": "keep"}]},  # stage 1
            {"needed": True, "concept": "contains molasses", "rationale": "..."},  # stage 2
            {"attributes": [new_attr]},  # stage 3
        ]
    )
    error_examples = {"false_positives": [{"a": 1}], "false_negatives": []}
    attrs, rationale = revise_attributes(
        client,
        "beans",
        [ATTR_A],
        correlation_flags=None,
        evaluation=None,
        error_examples=error_examples,
        sample_pairs=[],
        field_stats=None,
        candidate_terms=None,
        guidance=None,
        fndds_texts=[],
        off_texts=[],
        temperature=0.1,
    )
    assert ATTR_A in attrs
    assert new_attr in attrs
    assert "contains molasses" in rationale


def test_revise_attributes_no_op_rationale_when_nothing_changes():
    client = QueueClient(
        [
            {"decisions": [{"name": "beans_bean_type", "action": "keep"}]},
            {"needed": False},
        ]
    )
    attrs, rationale = revise_attributes(
        client,
        "beans",
        [ATTR_A],
        correlation_flags=None,
        evaluation=None,
        error_examples={"false_positives": [{"a": 1}], "false_negatives": []},
        sample_pairs=[],
        field_stats=None,
        candidate_terms=None,
        guidance=None,
        fndds_texts=[],
        off_texts=[],
        temperature=0.1,
    )
    assert attrs == [ATTR_A]
    assert rationale == "kept the attribute set unchanged"
