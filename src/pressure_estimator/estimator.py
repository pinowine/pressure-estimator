import json
from enum import Enum
from pathlib import Path

import numpy as np

from .input import PressureInput

class ExecutionDevice(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"

# device choice code come from in-class notebook
def resolve_execution_providers(device: ExecutionDevice | str, available_providers: list[str]) -> list[str]:
    selected = ExecutionDevice(device)
    available = set(available_providers)

    if selected is ExecutionDevice.cpu:
        if "CPUExecutionProvider" not in available:
            raise RuntimeError("CPUExecutionProvider is not available in this ONNX Runtime build.")
        return ["CPUExecutionProvider"]

    if selected is ExecutionDevice.cuda:
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Install the GPU runtime with "
                '`pip install -e ".[gpu]"` and make sure CUDA is visible to ONNX Runtime.'
            )

        providers = ["CUDAExecutionProvider"]
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")
        return providers

    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")

    if not providers:
        raise RuntimeError("No supported ONNX Runtime execution provider is available.")

    return providers

def load_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            'ONNX Runtime is not installed. Use `pip install -e ".[cpu]"` for CPU '
            'or `pip install -e ".[gpu]"` for CUDA.'
        ) from exc

    return ort

class PressureEstimator:
    def __init__(self, model_path: str, schema_path: str, device: ExecutionDevice | str = ExecutionDevice.auto):
        self.model_path = Path(model_path)
        self.schema_path = Path(schema_path)
        self.device = ExecutionDevice(device)

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with self.schema_path.open("r", encoding="utf-8") as f:
            self.schema = json.load(f)

        self.input_name = self.schema.get("modelInputName", "features")
        self.output_name = self.schema.get("modelOutputName", "pressure")
        # The schema owns feature order, keeping inference aligned with training/export.
        self.features = self.schema["features"]

        ort = load_onnxruntime()
        self.providers = resolve_execution_providers(self.device, ort.get_available_providers())

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # CUDA stays first when selected, CPU remains a practical fallback for unsupported ops, because I'm using CUDA though...
        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=self.providers,
        )
        self.active_providers = self.session.get_providers()

        if self.device is ExecutionDevice.cuda and "CUDAExecutionProvider" not in self.active_providers:
            raise RuntimeError("CUDA was requested, but the ONNX session did not enable CUDAExecutionProvider.")

    def predict(self, x: PressureInput) -> float:
        values: list[float] = []

        for feature in self.features:
            name = feature["name"]
            raw = getattr(x, name)

            mean = float(feature.get("mean", 0.0))
            scale = float(feature.get("scale", 1.0))

            # Some exported stats may have zero variance; treat them as already scaled.
            if abs(scale) < 1e-8:
                scale = 1.0

            values.append((raw - mean) / scale)

        arr = np.array([values], dtype=np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: arr},
        )

        pressure = float(outputs[0].reshape(-1)[0])
        # Pressure is a UI-facing score, so keep it inside the expected 0..1 range.
        return float(np.clip(pressure, 0.0, 1.0))

def pressure_band(value: float) -> str:
    if value < 0.25:
        return "Low"
    if value < 0.55:
        return "Normal"
    if value < 0.75:
        return "High"
    if value < 0.90:
        return "Critical"
    return "Mercy"
