from privacy_analyzer.cli import _extract_year


def test_four_digit_year():
    assert _extract_year("data/policies/fb_2010.txt") == "2010"
    assert _extract_year("fb_privacy_policy_2015.txt") == "2015"


def test_two_digit_year_normalized():
    assert _extract_year("tiktok_20.txt") == "2020"
    assert _extract_year("tiktok_21.txt") == "2021"


def test_no_digits_falls_back_to_stem():
    assert _extract_year("policy_draft.txt") == "policy_draft"
