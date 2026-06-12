from privacy_analyzer.diff import compute_section_diff
from privacy_analyzer.matching import embed_sections
from privacy_analyzer.segmentation import Section


def _sec(title, sentences):
    return Section(title=title, sentences=list(sentences))


def test_identical_sections_no_changes(encoder, cfg_path):
    sents = [
        "We collect your name and email address.",
        "We share information with trusted partners.",
    ]
    old, new = _sec("Data", sents), _sec("Data", sents)
    embed_sections([old, new], encoder)

    d = compute_section_diff(old, new, "2010", "2015", encoder, nlp=None, cfg_path=cfg_path)
    assert d.n_unchanged == 2
    assert d.n_added == 0
    assert d.n_removed == 0
    assert d.change_magnitude < 5


def test_added_and_removed_sentences(encoder, cfg_path):
    old = _sec("Data", ["We collect your name and email address."])
    new = _sec("Data", [
        "We collect your name and email address.",
        "Biometric faceprints voiceprints fingerprints retina iris scans.",
    ])
    embed_sections([old, new], encoder)

    d = compute_section_diff(old, new, "2010", "2015", encoder, nlp=None, cfg_path=cfg_path)
    assert d.n_unchanged == 1
    assert d.n_added == 1
    assert d.n_removed == 0
    assert d.change_magnitude > 0

    statuses = {c.text: c.status for c in d.sentence_changes}
    assert statuses["Biometric faceprints voiceprints fingerprints retina iris scans."] == "added"


def test_removed_sentence_detected(encoder, cfg_path):
    old = _sec("Data", [
        "We collect your name and email address.",
        "Anonymized aggregated statistics deleted after retention period expires.",
    ])
    new = _sec("Data", ["We collect your name and email address."])
    embed_sections([old, new], encoder)

    d = compute_section_diff(old, new, "2010", "2015", encoder, nlp=None, cfg_path=cfg_path)
    assert d.n_removed == 1
    removed = [c for c in d.sentence_changes if c.status == "removed"]
    assert removed[0].text.startswith("Anonymized")
