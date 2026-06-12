from privacy_analyzer.matching import build_chains, match_two
from privacy_analyzer.segmentation import Section


def _sec(title, sentences):
    return Section(title=title, sentences=list(sentences))


def test_identical_sections_match(encoder, cfg_path):
    old = [
        _sec("Data collection", ["We collect your name and email address."]),
        _sec("Data sharing", ["We share information with trusted partners."]),
    ]
    new = [
        _sec("Data sharing", ["We share information with trusted partners."]),
        _sec("Data collection", ["We collect your name and email address."]),
    ]
    matches = match_two(old, new, encoder, cfg_path)
    paired = {(m.old_section.title, m.new_section.title)
              for m in matches if not m.is_new and not m.is_removed}
    assert ("Data collection", "Data collection") in paired
    assert ("Data sharing", "Data sharing") in paired


def test_unmatched_section_is_new(encoder, cfg_path):
    old = [_sec("Data collection", ["We collect your name and email address."])]
    new = [
        _sec("Data collection", ["We collect your name and email address."]),
        _sec("Biometric identifiers", ["Faceprints voiceprints fingerprints retina scans iris."]),
    ]
    matches = match_two(old, new, encoder, cfg_path)
    new_titles = {m.new_section.title for m in matches if m.is_new}
    assert "Biometric identifiers" in new_titles


def test_build_chains_two_years(encoder, cfg_path):
    policies = {
        "2010": [_sec("Data collection", ["We collect your name and email address."])],
        "2015": [_sec("Data collection", ["We collect your name and email address."])],
    }
    chains = build_chains(policies, encoder, cfg_path)
    assert len(chains) == 1
    assert chains[0].sections["2010"] is not None
    assert chains[0].sections["2015"] is not None
    assert chains[0].similarities["2010→2015"] > 0.9
