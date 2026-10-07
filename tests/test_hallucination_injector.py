import pytest
from evaluation.benchmark.injector import HallucinationInjector

def test_append_claim():
    inj = HallucinationInjector()
    assert inj.append_claim("Base answer", "Bad claim.") == "Base answer. Bad claim."
    assert inj.append_claim("Base answer.", "Bad claim.") == "Base answer. Bad claim."
    assert inj.append_claim("", "Bad claim.") == "Bad claim."

def test_replace_phrase():
    inj = HallucinationInjector()
    assert inj.replace_phrase("A good day.", "good", "bad") == "A bad day."
    with pytest.raises(ValueError):
        inj.replace_phrase("A good day.", "missing", "bad")

def test_insert_claim():
    inj = HallucinationInjector()
    res = inj.insert_claim("A good day.", "very", "A ")
    assert res == "A  very good day."
    with pytest.raises(ValueError):
        inj.insert_claim("A good day.", "bad", "missing")
