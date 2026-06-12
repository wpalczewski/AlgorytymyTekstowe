import yaml

from privacy_analyzer.segmentation import (
    Section,
    _is_heading,
    _merge_small_sections,
    segment_strategy_a,
)

POLICY = """What information do we collect?

We collect information when you create an account and use the platform every day.
We also collect technical data about your device and your network connection details.

How do we use your information?

We use the collected information to operate and improve the platform for all users.
We may also use it to send you marketing communications about our products and services.

Who do we share information with?

We share your information with service providers that support our business operations globally.
We may also share data with law enforcement when required by applicable law in your country.
"""


def _cfg(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def test_is_heading():
    assert _is_heading("What information do we collect?", 80)
    assert _is_heading("3. Sharing information on Facebook.", 80)
    assert _is_heading("Information we collect:", 80)
    assert not _is_heading("We collect information when you create an account.", 80)
    assert not _is_heading("", 80)


def test_strategy_a_splits_on_headings(nlp, cfg_path):
    sections = segment_strategy_a(POLICY, nlp, _cfg(cfg_path))
    assert len(sections) == 3
    assert sections[0].title == "What information do we collect?"
    assert sections[2].title == "Who do we share information with?"
    assert all(sec.sentences for sec in sections)
    assert all(sec.strategy == "A" for sec in sections)


def test_merge_small_sections():
    sections = [
        Section(title="A", sentences=["s1", "s2", "s3"]),
        Section(title="B", sentences=["s4"]),  # za mała → scalona z A
        Section(title="C", sentences=["s5", "s6", "s7"]),
    ]
    merged = _merge_small_sections(sections, min_sents=3)
    assert len(merged) == 2
    assert merged[0].sentences == ["s1", "s2", "s3", "s4"]
