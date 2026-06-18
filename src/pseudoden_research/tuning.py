from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean

from .config import TelemetryConfig
from .geometry import Vec2
from .simulation import GameSimulation, _scripted_player_input
from .strategies import ModelTrainingReport, SklearnIncrementalStrategy


SELECTION_POLICY = (
    "recent_eval_accuracy - 0.5 * mean_abs_eval_delta - 0.25 * mean_prediction_change_rate "
    "- 0.01 * extra_sample_cost"
)

@dataclass(frozen=True)
class TuningCandidate:
    name: str
    phase: str
    rationale: str
    classifier_alpha: float = 0.0001
    classifier_learning_rate: str = "optimal"
    classifier_eta0: float = 0.01
    classifier_average: bool = True
    training_samples: int = 640
    evaluation_samples: int = 256
    classifier_loss: str = "log_loss"
    classifier_penalty: str = "l2"

DEFAULT_CANDIDATES = (
    TuningCandidate(
        name="baseline_current",
        phase="control",
        rationale="Use the current lightweight default selected after the first stability check.",
    ),
    TuningCandidate(
        name="previous_baseline",
        phase="historical_control",
        rationale="Keep the old non-averaged settings as a control for regression checks.",
        classifier_average=False,
    ),
    TuningCandidate(
        name="average_more_regularized",
        phase="stability",
        rationale="Use stronger regularization plus averaging because previous logs showed large prediction swings.",
        classifier_alpha=0.001,
        classifier_average=True,
    ),
    TuningCandidate(
        name="constant_lr_010_average",
        phase="learning_rate",
        rationale="Try a fixed small step size so each partial_fit changes the model less abruptly.",
        classifier_learning_rate="constant",
        classifier_eta0=0.01,
        classifier_average=True,
    ),
    TuningCandidate(
        name="constant_lr_005_average",
        phase="learning_rate",
        rationale="Use a smaller fixed step when 0.01 is still too jumpy.",
        classifier_learning_rate="constant",
        classifier_eta0=0.005,
        classifier_average=True,
    ),
    TuningCandidate(
        name="more_samples_average",
        phase="data",
        rationale="Increase teacher samples after stabilizing updates, so the model sees more path cases.",
        classifier_average=True,
        training_samples=1280,
    ),
)

def run_ml_tuning(
    episodes: int = 6,
    frames: int = 120,
    limit: int | None = None,
    dt: float = 1.0 / 60.0,
) -> dict[str, object]:
    episodes = max(1, episodes)
    frames = max(1, frames)
    candidates = DEFAULT_CANDIDATES[: max(1, limit)] if limit is not None else DEFAULT_CANDIDATES
    output_path = _tuning_log_path()
    run_id = output_path.stem.replace("ml_tuning_", "")
    model_root = Path("models") / "tuning" / run_id
    fieldnames = list(_empty_result_row().keys())
    rows: list[dict[str, object]] = []

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, candidate in enumerate(candidates, start=1):
            row = _run_candidate(candidate, index, episodes, frames, dt, model_root)
            writer.writerow(row)
            file.flush()
            rows.append(row)

    best_row = max(rows, key=lambda row: float(row["selection_score"]))
    return {
        "log": str(output_path),
        "candidates": len(rows),
        "best_candidate": best_row["candidate"],
        "best_score": best_row["selection_score"],
        "best_eval_accuracy": best_row["best_eval_accuracy"],
        "best_model_path": best_row["best_model_path"],
    }

def format_ml_tuning_summary(summary: dict[str, object]) -> str:
    return "\n".join(
        [
            "ML tuning complete",
            f"candidates: {summary['candidates']}",
            f"best_candidate: {summary['best_candidate']}",
            f"best_eval_accuracy: {summary['best_eval_accuracy']}",
            f"selection_score: {summary['best_score']}",
            f"best_model_path: {summary['best_model_path']}",
            f"log: {summary['log']}",
            f"selection_policy: {SELECTION_POLICY}",
        ]
    )

def _run_candidate(
    candidate: TuningCandidate,
    candidate_index: int,
    episodes: int,
    frames: int,
    dt: float,
    model_root: Path,
) -> dict[str, object]:
    candidate_dir = model_root / candidate.name
    model_path = candidate_dir / "model.joblib"
    best_model_path = candidate_dir / "best_model.joblib"
    best_metadata_path = candidate_dir / "best_model.json"
    eval_values: list[float] = []
    eval_deltas: list[float] = []
    prediction_changes: list[float] = []
    distances: list[float] = []
    saved_best_count = 0
    total_caught = 0
    last_report: ModelTrainingReport | None = None

    for episode in range(episodes):
        strategy = SklearnIncrementalStrategy(
            training_samples=candidate.training_samples,
            evaluation_samples=candidate.evaluation_samples,
            training_offset=episode * 997,
            classifier_loss=candidate.classifier_loss,
            classifier_penalty=candidate.classifier_penalty,
            classifier_alpha=candidate.classifier_alpha,
            classifier_learning_rate=candidate.classifier_learning_rate,
            classifier_eta0=candidate.classifier_eta0,
            classifier_average=candidate.classifier_average,
            model_path=model_path,
            best_model_path=best_model_path,
            best_metadata_path=best_metadata_path,
        )
        simulation = GameSimulation(strategy=strategy, telemetry_config=TelemetryConfig(enabled=False), seed=7 + episode)
        episode_distance = 0.0
        episode_caught = 0
        try:
            report = strategy.prepare_training(simulation.world)
            last_report = report
            _append_optional(eval_values, report.eval_accuracy_after)
            _append_optional_delta(eval_deltas, report.eval_accuracy_after, report.eval_accuracy_before)
            _append_optional(prediction_changes, report.eval_prediction_change_rate)
            saved_best_count += int(report.saved_best_model)

            for frame in range(frames):
                metrics = simulation.step(_scripted_player_input(simulation, frame, episode), dt)
                episode_distance += metrics.distance
                episode_caught += int(metrics.caught)
        finally:
            simulation.close()

        distances.append(episode_distance / frames)
        total_caught += episode_caught

    if last_report is None or not eval_values:
        raise RuntimeError(f"Tuning candidate {candidate.name} produced no evaluation data.")

    recent_window = min(3, len(eval_values))
    recent_eval = mean(eval_values[-recent_window:])
    mean_abs_delta = mean(abs(value) for value in eval_deltas) if eval_deltas else 0.0
    mean_change_rate = mean(prediction_changes) if prediction_changes else 0.0
    sample_cost_penalty = max(0.0, candidate.training_samples / 640 - 1.0) * 0.01
    selection_score = recent_eval - 0.5 * mean_abs_delta - 0.25 * mean_change_rate - sample_cost_penalty
    best_eval = max(eval_values)
    best_iteration = eval_values.index(best_eval) + 1

    row = _empty_result_row()
    row.update(
        {
            "candidate": candidate.name,
            "candidate_index": candidate_index,
            "phase": candidate.phase,
            "rationale": candidate.rationale,
            "episodes": episodes,
            "frames": frames,
            "classifier_loss": candidate.classifier_loss,
            "classifier_penalty": candidate.classifier_penalty,
            "classifier_alpha": candidate.classifier_alpha,
            "classifier_learning_rate": candidate.classifier_learning_rate,
            "classifier_eta0": candidate.classifier_eta0,
            "classifier_average": int(candidate.classifier_average),
            "feature_set": last_report.feature_set,
            "feature_count": last_report.feature_count,
            "training_samples": candidate.training_samples,
            "evaluation_samples": candidate.evaluation_samples,
            "first_eval_accuracy": _format_float(eval_values[0]),
            "last_eval_accuracy": _format_float(eval_values[-1]),
            "best_eval_accuracy": _format_float(best_eval),
            "best_iteration": best_iteration,
            "recent_eval_accuracy": _format_float(recent_eval),
            "mean_abs_eval_delta": _format_float(mean_abs_delta),
            "mean_prediction_change_rate": _format_float(mean_change_rate),
            "sample_cost_penalty": _format_float(sample_cost_penalty),
            "mean_avg_distance": _format_float(mean(distances)),
            "caught_total": total_caught,
            "saved_best_count": saved_best_count,
            "selection_score": _format_float(selection_score),
            "selection_policy": SELECTION_POLICY,
            "model_path": str(model_path),
            "best_model_path": last_report.best_model_path,
        }
    )
    return row

def _empty_result_row() -> dict[str, object]:
    return {
        "candidate": "",
        "candidate_index": "",
        "phase": "",
        "rationale": "",
        "episodes": "",
        "frames": "",
        "classifier_loss": "",
        "classifier_penalty": "",
        "classifier_alpha": "",
        "classifier_learning_rate": "",
        "classifier_eta0": "",
        "classifier_average": "",
        "feature_set": "",
        "feature_count": "",
        "training_samples": "",
        "evaluation_samples": "",
        "first_eval_accuracy": "",
        "last_eval_accuracy": "",
        "best_eval_accuracy": "",
        "best_iteration": "",
        "recent_eval_accuracy": "",
        "mean_abs_eval_delta": "",
        "mean_prediction_change_rate": "",
        "sample_cost_penalty": "",
        "mean_avg_distance": "",
        "caught_total": "",
        "saved_best_count": "",
        "selection_score": "",
        "selection_policy": "",
        "model_path": "",
        "best_model_path": "",
    }

def _append_optional(values: list[float], value: float | None) -> None:
    if value is not None:
        values.append(value)

def _append_optional_delta(values: list[float], after: float | None, before: float | None) -> None:
    if after is not None and before is not None:
        values.append(after - before)

def _format_float(value: float) -> str:
    return f"{value:.3f}"

def _tuning_log_path() -> Path:
    directory = Path("logs")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"ml_tuning_{stamp}.csv"
