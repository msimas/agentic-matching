from agentic_matching.llm.client import _strip_code_fence


def test_plain_json_passes_through_unchanged():
    assert _strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_strips_json_labeled_fence():
    content = '```json\n{"a": 1}\n```'
    assert _strip_code_fence(content) == '{"a": 1}'


def test_strips_bare_fence():
    content = '```\n{"a": 1}\n```'
    assert _strip_code_fence(content) == '{"a": 1}'


def test_strips_fence_with_surrounding_whitespace():
    content = '  \n```json\n{"a": 1}\n```\n  '
    assert _strip_code_fence(content) == '{"a": 1}'


def test_multiline_json_body_preserved():
    content = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
    assert _strip_code_fence(content) == '{\n  "a": 1,\n  "b": 2\n}'


def test_unfenced_content_with_backticks_inside_untouched():
    # No leading/trailing fence -- must not be mangled.
    content = 'not fenced ``` mid-string ``` still not fenced'
    assert _strip_code_fence(content) == content
