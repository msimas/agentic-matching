from agentic_matching.calibration import normalize_code


def test_strips_non_digits():
    assert normalize_code("0-12345-67890-5") == "12345678905"


def test_strips_leading_zeros():
    # EAN-13 is usually "0" + UPC-A; normalizing should make them comparable.
    assert normalize_code("0036000291452") == normalize_code("036000291452")


def test_none_input():
    assert normalize_code(None) is None


def test_empty_string():
    assert normalize_code("") is None


def test_all_zero_is_none():
    assert normalize_code("0000000000") is None


def test_no_digits_is_none():
    assert normalize_code("abc-def") is None


def test_upc_a_and_gtin_14_collapse_to_same_core():
    upc_a = "036000291452"
    gtin_14 = "00036000291452"  # zero-padded to 14 digits
    assert normalize_code(upc_a) == normalize_code(gtin_14)
