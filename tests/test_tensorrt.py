"""CPU checks for the TensorRT shape and parity contract; no GPU packages needed."""

import subprocess
import sys

import numpy as np
import pytest

from deploy.export_tensorrt import compare_relation_outputs, input_profiles


@pytest.fixture
def bank(tmp_path):
    path = tmp_path / "predicate_bank.npz"
    np.savez(
        path,
        W=np.ones((7, 8), np.float32),
        default=np.array(["on", "beside", "riding"]),
    )
    return path


def test_profiles_keep_vocabulary_dynamic_and_regions_fixed(bank):
    meta = dict(img_size=448, max_boxes=32, vocab_mode="input", text_dim=8)
    shapes = input_profiles(meta, bank)
    assert shapes["image"] == ((1, 3, 448, 448),) * 3
    assert shapes["boxes"] == ((1, 32, 4),) * 3
    assert shapes["box_counts"] == ((1,),) * 3
    assert shapes["W"] == ((1, 8), (3, 8), (7, 8))
    assert shapes["alpha"] == ((1,), (3,), (7,))
    assert input_profiles(meta, bank, max_vocab=2)["W"] == ((1, 8), (2, 8), (2, 8))
    assert input_profiles(meta, bank, max_vocab=20)["alpha"][-1] == (20,)


def test_baked_and_detector_profiles():
    assert set(input_profiles(dict(img_size=448, max_boxes=32))) == {
        "image",
        "boxes",
        "box_counts",
    }
    assert input_profiles(dict(imgsz=640)) == {"images": ((1, 3, 640, 640),) * 3}


def test_invalid_bank_and_profile_fail_early(bank):
    meta = dict(img_size=448, max_boxes=32, vocab_mode="input", text_dim=8)
    with pytest.raises(ValueError, match="predicate_bank"):
        input_profiles(meta)
    with pytest.raises(ValueError, match="positive"):
        input_profiles(meta, bank, max_vocab=0)
    with pytest.raises(ValueError, match="dimension"):
        input_profiles(dict(meta, text_dim=12), bank)


def outputs():
    return [
        np.array([[[2.0, 3.0], [5.0, 6.0], [99.0, 99.0]]], np.float32),
        np.array([[0.5, 1.5, 99.0]], np.float32),
        np.array([[0, 1, 0]], np.int64),
        np.array([[1, 0, 0]], np.int64),
        np.array([[True, True, False]]),
    ]


def test_parity_aligns_pairs_and_ignores_invalid_padding():
    ref = outputs()
    got = [x[:, [2, 1, 0]].copy() for x in ref]
    got[0][0, 0] = -10000
    got[2][0, 0] = 30
    assert compare_relation_outputs(ref, got) == 0


@pytest.mark.parametrize(
    "change", ["missing", "different", "duplicate", "logits", "pair_logits", "nan"]
)
def test_parity_rejects_incorrect_predictions(change):
    ref, got = outputs(), outputs()
    if change == "missing":
        got[4][0, 0] = False
    elif change == "different":
        got[3][0, 0] = 2
    elif change == "duplicate":
        got[2][0, 1], got[3][0, 1] = 0, 1
    elif change == "logits":
        got[0][0, 0, 0] += 0.1
    elif change == "pair_logits":
        got[1][0, 0] += 0.1
    else:
        got[0][0, 0, 0] = np.nan
    with pytest.raises(AssertionError):
        compare_relation_outputs(ref, got)


def test_empty_pair_set():
    ref, got = outputs(), outputs()
    ref[-1][:] = False
    got[-1][:] = False
    assert compare_relation_outputs(ref, got) == 0


def test_backend_rejects_typo():
    from deploy.runtime import ScenePipeline

    with pytest.raises(ValueError, match="Unknown backend"):
        ScenePipeline(backend="tensortr")


def test_gpu_dependencies_are_optional_at_import():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules.update(torch=None, tensorrt=None, onnxruntime=None); "
            "import deploy.trt_runtime, deploy.export_tensorrt",
        ],
        check=True,
    )
