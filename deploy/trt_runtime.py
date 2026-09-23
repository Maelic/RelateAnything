"""Native TensorRT 10 runtime, sharing the ONNX pre/postprocessing contract.

PyTorch is used only for CUDA memory and streams; model execution is entirely
TensorRT. Imports are lazy so CPU/ONNX deployments need neither dependency.
Engines are built locally with deploy/export_tensorrt.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deploy.runtime import OnnxDetector, OnnxRelationHead


def load_tensorrt():
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ImportError(
            "TensorRT requires NVIDIA's TensorRT 10 Python package. "
            "Install with pip install -e '.[tensorrt]' (see deploy/README.md)."
        ) from exc
    if int(trt.__version__.split(".")[0]) != 10:
        raise RuntimeError(f"TensorRT 10.x is required; found {trt.__version__}")
    return trt


class TensorRTEngine:
    """Synchronous numpy interface to one engine / execution context.

    One instance must not be used concurrently. Buffers are reused until their
    shapes change; returned arrays own their data and survive subsequent calls.
    """

    def __init__(self, path: str, device: str = "cuda"):
        import torch

        self.torch = torch
        self.trt = trt = load_tensorrt()
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError(
                "TensorRT needs an NVIDIA GPU and CUDA-enabled PyTorch; use device='cuda'."
            )
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"{path}: build the engine with deploy/export_tensorrt.py"
            )
        self.logger = trt.Logger(trt.Logger.WARNING)
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise RuntimeError(
                    f"Cannot load {path}; rebuild it on this GPU with the installed TensorRT version."
                )
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError(
                    f"Cannot create TensorRT execution context for {path}"
                )
            self.stream = torch.cuda.Stream(device=self.device)
        names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        self.input_names = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        ]
        self.output_names = [n for n in names if n not in self.input_names]
        self.dtypes = {
            n: np.dtype(trt.nptype(self.engine.get_tensor_dtype(n))) for n in names
        }
        for name in names:
            if self.engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise ValueError(
                    f"Host tensor {name!r} is unsupported; rebuild with export_tensorrt.py"
                )
        self.buffers = {}

    def _buffer(self, name, shape):
        torch = self.torch
        dtype = torch.from_numpy(np.empty((), dtype=self.dtypes[name])).dtype
        current = self.buffers.get(name)
        if current is None or tuple(current.shape) != tuple(shape):
            current = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            self.buffers[name] = current
        if not self.context.set_tensor_address(name, current.data_ptr()):
            raise RuntimeError(f"Cannot bind TensorRT tensor {name!r}")
        return current

    def run(self, feed: dict, output_names=None):
        if set(feed) != set(self.input_names):
            raise ValueError(f"Expected inputs {self.input_names}, got {list(feed)}")
        names = self.output_names if output_names is None else list(output_names)
        if not set(names) <= set(self.output_names):
            raise ValueError(f"Unknown output names: {names}")
        arrays = {}
        for name in self.input_names:
            value = np.asarray(feed[name])
            if value.dtype != self.dtypes[name]:
                raise ValueError(
                    f"{name}: expected {self.dtypes[name]}, got {value.dtype}"
                )
            shape = tuple(value.shape)
            expected = tuple(self.engine.get_tensor_shape(name))
            if len(shape) != len(expected) or any(
                d >= 0 and d != s for d, s in zip(expected, shape)
            ):
                raise ValueError(f"{name}: expected shape {expected}, got {shape}")
            if -1 in expected:
                low, _, high = self.engine.get_tensor_profile_shape(name, 0)
                if any(s < lo or s > hi for s, lo, hi in zip(shape, low, high)):
                    raise ValueError(
                        f"{name}: shape {shape} outside engine profile {tuple(low)}..{tuple(high)}; rebuild the engine"
                    )
            arrays[name] = np.ascontiguousarray(value)
        if (
            "W" in arrays
            and "alpha" in arrays
            and len(arrays["W"]) != len(arrays["alpha"])
        ):
            raise ValueError("W and alpha must have the same predicate count")

        torch = self.torch
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            # Synchronize even on an exception before releasing temporary host
            # buffers or allowing a caller to reuse this execution context.
            try:
                for name, value in arrays.items():
                    if not self.context.set_input_shape(name, value.shape):
                        raise ValueError(
                            f"TensorRT rejected {name} shape {value.shape}"
                        )
                    self._buffer(name, value.shape).copy_(torch.from_numpy(value))
                missing = self.context.infer_shapes()
                if missing:
                    raise RuntimeError(f"TensorRT cannot infer shapes: {missing}")
                for name in self.output_names:
                    shape = tuple(self.context.get_tensor_shape(name))
                    if any(d < 0 for d in shape):
                        raise RuntimeError(
                            f"Unresolved output shape for {name}: {shape}"
                        )
                    self._buffer(name, shape)
                if not self.context.execute_async_v3(self.stream.cuda_stream):
                    raise RuntimeError("TensorRT execution failed")
                outputs = [self.buffers[n].cpu().numpy() for n in names]
            finally:
                self.stream.synchronize()
        return outputs


class TensorRTDetector(OnnxDetector):
    def __init__(self, engine_path: str, device: str = "cuda", **kwargs):
        self._device = device
        super().__init__(engine_path, **kwargs)

    def _build_engine(self, path, threads, providers):
        self.engine = TensorRTEngine(path, self._device)
        if len(self.engine.input_names) != 1 or len(self.engine.output_names) != 1:
            raise ValueError(
                "Expected a raw YOLO detector engine with one input and one output"
            )
        self.iname = self.engine.input_names[0]

    def _run_engine(self, x):
        return self.engine.run({self.iname: x})[0]


class TensorRTRelationHead(OnnxRelationHead):
    def __init__(
        self, engine_path: str, bank_path: str = "", device: str = "cuda", **kwargs
    ):
        self._device = device
        super().__init__(engine_path, bank_path=bank_path, **kwargs)

    def _build_engine(self, path, threads, providers):
        self.engine = TensorRTEngine(path, self._device)
        self.input_names = self.engine.input_names
        # TensorRT may reorder outputs. Bind by semantic name, never position.
        self.output_names = [
            "pred_logits",
            "pair_logits",
            "sub_idx",
            "obj_idx",
            "valid_mask",
        ]
        if set(self.engine.output_names) != set(self.output_names):
            raise ValueError(
                "TensorRT requires a v2 logits graph; re-export with deploy/export_onnx.py"
            )
        return self.output_names

    def _run_engine(self, feed):
        return self.engine.run(feed, self.output_names)
