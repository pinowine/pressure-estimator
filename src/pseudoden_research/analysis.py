from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


STAGE_ONE_TARGET_EVAL = 0.65
MIN_ROWS_FOR_TREND = 10
PLATEAU_DELTA = 0.01
UNSTABLE_DELTA = 0.06
UNSTABLE_CHANGE_RATE = 0.5


def analyze_ml_training_log(path: str | Path | None = None) -> dict[str, object]:
    log_path = _resolve_log_path(path)
    rows = _read_rows(log_path)
    if not rows:
        raise ValueError(f"No rows found in {log_path}")

    eval_pairs = _numbered_values(rows, "eval_accuracy_after")
    if not eval_pairs:
        raise ValueError(f"No eval_accuracy_after values found in {log_path}")

    eval_values = [value for _, value in eval_pairs]
    first_eval = eval_values[0]
    last_eval = eval_values[-1]
    best_row, best_eval = max(eval_pairs, key=lambda item: item[1])
    worst_row, worst_eval = min(eval_pairs, key=lambda item: item[1])
    window = min(5, len(eval_values))
    recent_avg = mean(eval_values[-window:])
    previous_avg = mean(eval_values[-window * 2 : -window]) if len(eval_values) >= window * 2 else None
    eval_deltas = [abs(value) for value in _float_values(rows, "eval_accuracy_delta")]
    prediction_changes = _float_values(rows, "eval_prediction_change_rate")
    avg_distances = _float_values(rows, "avg_distance")

    summary = {
        "log": str(log_path),
        "rows": len(rows),
        "first_eval_accuracy": first_eval,
        "last_eval_accuracy": last_eval,
        "best_eval_accuracy": best_eval,
        "best_iteration": _row_iteration(rows[best_row - 1], best_row),
        "worst_eval_accuracy": worst_eval,
        "worst_iteration": _row_iteration(rows[worst_row - 1], worst_row),
        "mean_eval_accuracy": mean(eval_values),
        "recent_window": window,
        "recent_eval_accuracy": recent_avg,
        "previous_eval_accuracy": previous_avg,
        "recent_eval_delta": None if previous_avg is None else recent_avg - previous_avg,
        "mean_abs_eval_delta": mean(eval_deltas) if eval_deltas else None,
        "mean_prediction_change_rate": mean(prediction_changes) if prediction_changes else None,
        "mean_avg_distance": mean(avg_distances) if avg_distances else None,
        "caught_total": sum(_int(row.get("caught")) for row in rows),
        "saved_best_count": sum(_int(row.get("saved_best_model")) for row in rows),
        "best_model_path": _last_text(rows, "best_model_path"),
    }
    summary["training_state"] = _training_state(summary)
    summary["recommendations"] = _recommend(summary)
    return summary

def format_ml_training_analysis(summary: dict[str, object]) -> str:
    lines = [
        "ML training analysis",
        f"log: {summary['log']}",
        f"rows: {summary['rows']}",
        "",
        "accuracy",
        f"first_eval_accuracy: {_fmt(summary['first_eval_accuracy'])}",
        f"last_eval_accuracy: {_fmt(summary['last_eval_accuracy'])}",
        f"best_eval_accuracy: {_fmt(summary['best_eval_accuracy'])} at iteration {summary['best_iteration']}",
        f"worst_eval_accuracy: {_fmt(summary['worst_eval_accuracy'])} at iteration {summary['worst_iteration']}",
        f"mean_eval_accuracy: {_fmt(summary['mean_eval_accuracy'])}",
        f"recent_{summary['recent_window']}_avg: {_fmt(summary['recent_eval_accuracy'])}",
        f"recent_delta_vs_previous: {_fmt(summary['recent_eval_delta'])}",
        "",
        "stability",
        f"mean_abs_eval_delta: {_fmt(summary['mean_abs_eval_delta'])}",
        f"mean_prediction_change_rate: {_fmt(summary['mean_prediction_change_rate'])}",
        "",
        "gameplay",
        f"mean_avg_distance: {_fmt(summary['mean_avg_distance'])}",
        f"caught_total: {summary['caught_total']}",
        "",
        "best model",
        f"saved_best_count: {summary['saved_best_count']}",
        f"best_model_path: {summary['best_model_path'] or 'n/a'}",
        "",
        "status",
        f"state: {summary['training_state']['state']}",
        f"decision: {summary['training_state']['decision']}",
        f"reason: {summary['training_state']['reason']}",
        "",
        "next",
    ]
    lines.extend(f"- {item}" for item in summary["recommendations"])  # type: ignore[index]
    return "\n".join(lines)

def _resolve_log_path(path: str | Path | None) -> Path:
    if path and str(path) != "latest":
        return Path(path)
    logs = sorted(Path("logs").glob("headless_ml_train_*.csv"), key=lambda item: item.stat().st_mtime)
    if not logs:
        raise FileNotFoundError("No headless ML training logs found under logs/")
    return logs[-1]

def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))

def _numbered_values(rows: list[dict[str, str]], key: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for index, row in enumerate(rows, start=1):
        value = _float(row.get(key))
        if value is not None:
            values.append((index, value))
    return values

def _float_values(rows: list[dict[str, str]], key: str) -> list[float]:
    return [value for row in rows if (value := _float(row.get(key))) is not None]

def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None

def _int(value: str | None) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0

def _row_iteration(row: dict[str, str], fallback: int) -> int:
    return _int(row.get("iteration")) or fallback

def _last_text(rows: list[dict[str, str]], key: str) -> str:
    for row in reversed(rows):
        value = row.get(key, "")
        if value:
            return value
    return ""

def _recommend(summary: dict[str, object]) -> list[str]:
    rows = int(summary["rows"])
    best = float(summary["best_eval_accuracy"])
    last = float(summary["last_eval_accuracy"])
    mean_change = summary["mean_prediction_change_rate"]
    volatility = summary["mean_abs_eval_delta"]
    recent_delta = summary["recent_eval_delta"]
    recommendations: list[str] = []

    # if rows < 10:
    #     recommendations.append("Run at least 10-20 iterations before trusting the trend.")
    if last < best - 0.03:
        recommendations.append("Use the saved best model for comparison, because the latest model is below the peak.")
    if isinstance(volatility, float) and volatility > 0.06:
        recommendations.append("Training is unstable; keep best-model saving on and improve features before long runs.")
    if isinstance(mean_change, float) and mean_change > 0.5:
        recommendations.append("Predictions are changing a lot; lower update noise or use a stronger feature set.")
    if best < 0.65:
        recommendations.append("The current feature set is probably too small; add richer map and distance features next.")
    if isinstance(recent_delta, float) and recent_delta > 0.02:
        recommendations.append("Recent iterations are improving; run a short confirmation batch with the same settings.")
    if not recommendations:
        recommendations.append("The model looks stable enough for an A* vs ML comparison run.")
    return recommendations

def _training_state(summary: dict[str, object]) -> dict[str, str]:
    rows = int(summary["rows"])
    best = float(summary["best_eval_accuracy"])
    recent_delta = _as_float(summary["recent_eval_delta"])
    volatility = _as_float(summary["mean_abs_eval_delta"])
    change_rate = _as_float(summary["mean_prediction_change_rate"])

    # small logs are useful for smoke checks, but not enough for a trend call
    if rows < MIN_ROWS_FOR_TREND:
        return _state("collecting_data", "continue_training", "not enough rows for a stable trend yet")
    # high accuracy is only complete when the model is also no longer moving much
    if best >= STAGE_ONE_TARGET_EVAL and _abs_under(recent_delta, PLATEAU_DELTA) and _under(change_rate, 0.2):
        return _state("complete", "run_astar_vs_ml", "stage-one target is met and recent updates are stable")
    # large swings mean more training alone is probably not the right fix
    if _over(volatility, UNSTABLE_DELTA) or _over(change_rate, UNSTABLE_CHANGE_RATE):
        return _state("unstable", "adjust_features_or_params", "accuracy or predictions are still moving too much")
    # flat accuracy below target usually means the current feature set is too weak
    if best < STAGE_ONE_TARGET_EVAL and _abs_under(recent_delta, PLATEAU_DELTA):
        return _state("needs_features", "add_features", "accuracy has plateaued below the stage-one target")
    if _over(recent_delta, PLATEAU_DELTA):
        return _state("improving", "continue_training", "recent average accuracy is still rising")
    return _state("plateau", "inspect_best_model", "training is not clearly improving")

def _state(state: str, decision: str, reason: str) -> dict[str, str]:
    return {"state": state, "decision": decision, "reason": reason}

def _as_float(value: object) -> float | None:
    return value if isinstance(value, float) else None

def _under(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold

def _over(value: float | None, threshold: float) -> bool:
    return value is not None and value > threshold

def _abs_under(value: float | None, threshold: float) -> bool:
    return value is not None and abs(value) <= threshold

def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    if value is None:
        return "n/a"
    return str(value)
