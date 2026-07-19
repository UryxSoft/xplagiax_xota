"""
[Fase-2 M-16 / audit C-07] Checkpoint smoke test.

The two seed checkpoint directories are named `Model_groups_3class_*` but every model is
loaded with num_labels=41. If any checkpoint actually had a 3-class head, load_state_dict
would blow up (or worse, silently mis-map). This test loads the real weights and asserts
the classifier heads match label_mapping.

Loading ~570 MB × 3 of weights is slow, so the test is opt-in:
    RUN_MODEL_SMOKE=1 pytest tests/test_model_shapes.py
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "engine"))

_ENGINE_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "engine")
_WEIGHTS = [
    os.path.join(_ENGINE_DIR, "modernbert.bin"),
    os.path.join(_ENGINE_DIR, "Model_groups_3class_seed12"),
    os.path.join(_ENGINE_DIR, "Model_groups_3class_seed22"),
]

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_MODEL_SMOKE") != "1" or not all(os.path.exists(p) for p in _WEIGHTS),
    reason="model smoke test is opt-in (RUN_MODEL_SMOKE=1) and requires local weights",
)


def test_all_heads_are_41_class():
    import detector_final as df

    assert len(df.label_mapping) == 41
    assert df.label_mapping[24] == "human"
    for name, model in (("model_1", df.model_1), ("model_2", df.model_2),
                        ("model_3", df.model_3)):
        out_features = model.classifier.out_features
        assert out_features == 41, f"{name}: head has {out_features} classes, expected 41"
