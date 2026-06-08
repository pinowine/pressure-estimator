import json
from pathlib import Path

import typer

from .estimator import ExecutionDevice, PressureEstimator, pressure_band
from .input import PressureInput

app = typer.Typer(help="Pressure Estimator CLI")


@app.command()
def predict(
    input_json: str = typer.Option("sample_input.json", help="Path to input json."),
    model: str = typer.Option("models/pressure_model.onnx", help="Path to ONNX model."),
    schema: str = typer.Option("models/pressure_feature_schema.json", help="Path to feature schema json."),
    device: ExecutionDevice = typer.Option(
        ExecutionDevice.auto,
        help="Inference device. Auto prefers CUDA when available.",
    ),
):
    input_path = Path(input_json)

    if not input_path.exists():
        raise FileNotFoundError(f"Input json not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    estimator = PressureEstimator(model_path=model, schema_path=schema, device=device)
    # Keep CLI input close to the game export; Pydantic handles the contract check.
    pressure = estimator.predict(PressureInput(**data))

    result = {
        "pressure": pressure,
        "band": pressure_band(pressure),
        "source": "onnx",
        "device": device.value,
        "providers": estimator.active_providers,
    }

    typer.echo(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
