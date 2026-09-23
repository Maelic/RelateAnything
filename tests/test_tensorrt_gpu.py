"""Optional CUDA integration test, skipped on the CPU-only CI runners."""

import numpy as np
import pytest


def test_sparse_sampler_onnx_cuda_preserves_unique_valid_pairs(tmp_path):
    """Regression for CUDA TopK duplicating padding at the dtype minimum."""
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    if (
        not torch.cuda.is_available()
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        pytest.skip("requires CUDA ONNX Runtime")
    from relsgg.model.sampler import RelatednessPairSampler

    ort.preload_dlls()
    torch.manual_seed(42)
    sampler = RelatednessPairSampler(feat_dim=8, rel_dim=4).eval()
    boxes = torch.rand(1, 32, 4) * 0.5 + 0.25
    feats = torch.randn(1, 32, 8)
    counts = torch.tensor([2])
    path = tmp_path / "sampler.onnx"
    with torch.inference_mode():
        torch.onnx.export(
            sampler,
            (boxes, feats, counts),
            str(path),
            input_names=["boxes", "feats", "counts"],
            opset_version=17,
            dynamo=False,
        )
    session = ort.InferenceSession(
        str(path), providers=[("CUDAExecutionProvider", {"use_tf32": 0})]
    )
    assert "CUDAExecutionProvider" in session.get_providers()
    for count in (0, 1, 2, 3, 7, 12, 20, 32):
        counts[0] = count
        with torch.inference_mode():
            reference = [x.numpy() for x in sampler(boxes, feats, counts)]
        actual = session.run(
            None,
            {"boxes": boxes.numpy(), "feats": feats.numpy(), "counts": counts.numpy()},
        )

        def valid_pairs(outputs):
            sub, obj, valid = outputs[:3]
            pairs = list(zip(sub[valid], obj[valid]))
            assert len(pairs) == len(set(pairs)) == min(128, count * (count - 1))
            return {pair: score for pair, score in zip(pairs, outputs[-1][valid])}

        ref, got = valid_pairs(reference), valid_pairs(actual)
        assert ref.keys() == got.keys()
        np.testing.assert_allclose(
            [got[key] for key in ref], list(ref.values()), atol=1e-5, rtol=1e-4
        )


def test_native_engine_dynamic_shapes_output_order_and_buffer_lifetime(tmp_path):
    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or int(trt.__version__.split(".")[0]) != 10:
        pytest.skip("requires an NVIDIA GPU and TensorRT 10")
    from deploy.trt_runtime import TensorRTEngine

    logger = trt.Logger(trt.Logger.ERROR)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    )
    x = network.add_input("x", trt.float32, (-1, 2))
    one = network.add_constant((1, 2), np.ones((1, 2), np.float32)).get_output(0)
    y = network.add_elementwise(x, one, trt.ElementWiseOperation.SUM).get_output(0)
    y.name = "plus_one"
    z = network.add_elementwise(x, one, trt.ElementWiseOperation.SUB).get_output(0)
    z.name = "minus_one"
    network.mark_output(z)
    network.mark_output(y)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 2), (3, 2), (7, 2))
    config.add_optimization_profile(profile)
    plan = builder.build_serialized_network(network, config)
    assert plan is not None
    path = tmp_path / "tiny.engine"
    path.write_bytes(bytes(plan))
    engine = TensorRTEngine(str(path))
    saved = []
    for rows in (1, 7, 3, 1):
        # Deliberately non-contiguous host input.
        data = np.arange(rows * 4, dtype=np.float32).reshape(rows, 4)[:, ::2]
        plus, minus = engine.run({"x": data}, ["plus_one", "minus_one"])
        np.testing.assert_array_equal(plus, data + 1)
        np.testing.assert_array_equal(minus, data - 1)
        saved.append((plus, data + 1))
    for result, expected in saved:
        np.testing.assert_array_equal(result, expected)
    for feed, message in [
        ({}, "Expected inputs"),
        ({"x": np.zeros((8, 2), np.float32)}, "outside engine profile"),
        ({"x": np.zeros((1, 3), np.float32)}, "expected shape"),
        ({"x": np.zeros((1, 2), np.float64)}, "expected float32"),
    ]:
        with pytest.raises(ValueError, match=message):
            engine.run(feed)


def test_detector_export_sidecar_and_shared_preprocessing(tmp_path):
    import json

    trt = pytest.importorskip("tensorrt")
    torch = pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    if not torch.cuda.is_available() or int(trt.__version__.split(".")[0]) != 10:
        pytest.skip("requires an NVIDIA GPU and TensorRT 10")
    from deploy.export_tensorrt import build_engine, check_engine
    from deploy.runtime import DetectorConfig
    from deploy.trt_runtime import TensorRTDetector

    h = onnx.helper
    raw = np.array(
        [[[2, 6], [2, 6], [2, 2], [2, 2], [0.9, 0.1], [0.1, 0.9]]], np.float32
    )
    graph = h.make_graph(
        [
            h.make_node("ReduceMean", ["images"], ["mean"], keepdims=0),
            h.make_node("Add", ["raw", "mean"], ["detections"]),
        ],
        "tiny_detector",
        [h.make_tensor_value_info("images", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [h.make_tensor_value_info("detections", onnx.TensorProto.FLOAT, [1, 6, 2])],
        [onnx.numpy_helper.from_array(raw, name="raw")],
    )
    source = tmp_path / "detector.onnx"
    onnx.save(
        h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=8),
        source,
    )
    metadata = {"imgsz": 8, "classes": ["cat", "dog"], "layout": "yolov8_raw_cxcywh"}
    original = json.dumps(metadata)
    source.with_suffix(".json").write_text(original)
    engine = build_engine(source)
    assert source.with_suffix(".json").read_text() == original
    report = check_engine(source, engine)
    assert report["cases"] == 1
    saved = json.loads(engine.with_suffix(".json").read_text())
    assert saved["classes"] == metadata["classes"]
    assert saved["tensorrt"]["onnx_parity"] == report
    detector = TensorRTDetector(str(engine))
    boxes, scores, labels = detector(np.zeros((8, 8, 3), np.uint8), DetectorConfig())
    assert set(labels) == {"cat", "dog"}
    for box, score, label in zip(boxes, scores, labels):
        np.testing.assert_allclose(
            box, [1, 1, 3, 3] if label == "cat" else [5, 5, 7, 7]
        )
        assert score == pytest.approx(0.9)
