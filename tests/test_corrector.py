import pytest
from correction.corrector import BaseCorrector

def test_base_corrector_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseCorrector()

def test_base_corrector_subclass():
    class DummyCorrector(BaseCorrector):
        def correct(self, o, c, e, v):
            return {}
    
    dc = DummyCorrector()
    assert dc.correct(None, None, None, None) == {}
