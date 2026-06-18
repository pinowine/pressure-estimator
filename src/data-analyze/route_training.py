from __future__ import annotations

import csv
import json
import random
import sys
import argparse
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
SRC = HERE.parent
ROOT = SRC.parent
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from joblib import dump, load
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier

from processor import _route_input
from pseudoden_research.config import TelemetryConfig, WorldConfig
from pseudoden_research.simulation import GameSimulation
from pseudoden_research.strategies import AStarStrategy, FEATURE_SET_NAME, MOVE_LABELS, SklearnIncrementalStrategy
from pseudoden_research.world import WorldState


# --- this part comes from ---
# concept ref: https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0/
# api refs: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
# api refs: https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html
# route collection and promotion gates are project-specific experiment automation

TRAIN_ROUTES = ("box", "zigzag", "orbit")
TEST_ROUTES = ("diagonal_sweep",)

def run_route_training(
    episodes: int = 12,
    frames: int = 600,
    passes: int = 3,
    train_routes: tuple[str, ...] = TRAIN_ROUTES,
    test_routes: tuple[str, ...] = TEST_ROUTES,
    random_samples: int = 512,
) -> dict[str, object]:
    # train on seen routes, then gate promotion with held-out route and random eval
    episodes = max(1, episodes)
    frames = max(1, frames)
    passes = max(1, passes)
    validation_episodes = max(2, episodes // 3)
    strategy = SklearnIncrementalStrategy()
    world = WorldState(WorldConfig())
    log_path = _route_training_log_path()
    route_model_path = ROOT / "models" / "route_imitation_sgd.joblib"
    route_best_path = ROOT / "models" / "route_best_imitation_sgd.joblib"
    route_deploy_path = ROOT / "models" / "route_deploy_imitation_sgd.joblib"
    route_metadata_path = ROOT / "models" / "route_best_imitation_sgd.json"
    route_deploy_metadata_path = ROOT / "models" / "route_deploy_imitation_sgd.json"

    # random samples stop route data from becoming the whole world
    train_x, train_y = _collect_samples(strategy, train_routes, episodes, frames, 0)
    random_x, random_y = strategy._build_teacher_dataset(world, random_samples, 40000)
    train_x.extend(random_x)
    train_y.extend(random_y)
    validation_x, validation_y = _collect_samples(strategy, train_routes, validation_episodes, frames, episodes + 100)
    test_x, test_y = _collect_samples(strategy, test_routes, episodes, frames, 1000)
    random_eval_x, random_eval_y = strategy._build_teacher_dataset(world, max(256, random_samples // 2), 50000)
    _ensure_samples(train_x, validation_x, test_x, random_eval_x)

    model = _load_or_create_model(strategy, world)
    baseline = _score_sets(model, train_x, train_y, validation_x, validation_y, test_x, test_y, random_eval_x, random_eval_y)
    best_validation = baseline["validation_accuracy"]
    best_test = baseline["test_accuracy"]
    best_random = baseline["random_accuracy"]
    best_iteration = 0
    deploy_validation = baseline["validation_accuracy"]
    deploy_test = baseline["test_accuracy"]
    deploy_iteration = 0
    promoted = False
    fieldnames = [
        "iteration",
        "train_routes",
        "validation_routes",
        "test_routes",
        "train_samples",
        "validation_samples",
        "test_samples",
        "random_eval_samples",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "random_accuracy",
        "validation_delta",
        "test_delta",
        "saved_route_best",
        "saved_deployable",
        "promoted_active",
        "route_model_path",
        "route_best_model_path",
        "route_deploy_model_path",
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            _log_row(
                0,
                train_routes,
                test_routes,
                train_x,
                validation_x,
                test_x,
                random_eval_x,
                baseline,
                baseline,
                False,
                False,
                False,
                route_model_path,
                route_best_path,
                route_deploy_path,
            )
        )
        for iteration in range(1, passes + 1):
            features, labels = _shuffled(train_x, train_y, seed=iteration)
            model.partial_fit(features, labels, classes=list(range(len(MOVE_LABELS))))  # type: ignore[attr-defined]
            scores = _score_sets(model, train_x, train_y, validation_x, validation_y, test_x, test_y, random_eval_x, random_eval_y)
            route_model_path.parent.mkdir(parents=True, exist_ok=True)
            dump(model, route_model_path)
            saved_best = scores["validation_accuracy"] > best_validation
            if saved_best:
                # route_best tracks the strongest route-only score
                best_validation = scores["validation_accuracy"]
                best_test = scores["test_accuracy"]
                best_random = scores["random_accuracy"]
                best_iteration = iteration
                dump(model, route_best_path)
                _write_route_metadata(route_metadata_path, scores, train_routes, test_routes, iteration, route_best_path)
            saved_deployable = _is_deployable(scores, baseline) and scores["validation_accuracy"] > deploy_validation
            if saved_deployable:
                # deployable is the safer candidate for the live model
                deploy_validation = scores["validation_accuracy"]
                deploy_test = scores["test_accuracy"]
                deploy_iteration = iteration
                dump(model, route_deploy_path)
                _write_route_metadata(
                    route_deploy_metadata_path,
                    scores,
                    train_routes,
                    test_routes,
                    iteration,
                    route_deploy_path,
                )
            writer.writerow(
                _log_row(
                    iteration,
                    train_routes,
                    test_routes,
                    train_x,
                    validation_x,
                    test_x,
                    random_eval_x,
                    scores,
                    baseline,
                    saved_best,
                    saved_deployable,
                    False,
                    route_model_path,
                    route_best_path,
                    route_deploy_path,
                )
            )

    if deploy_iteration > 0:
        promoted = _promote_route_model(strategy, route_deploy_path, route_deploy_metadata_path)
        _mark_promoted(log_path, deploy_iteration)

    return {
        "log": str(log_path),
        "train_routes": ",".join(train_routes),
        "test_routes": ",".join(test_routes),
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "test_samples": len(test_x),
        "baseline_validation_accuracy": baseline["validation_accuracy"],
        "best_validation_accuracy": best_validation,
        "baseline_test_accuracy": baseline["test_accuracy"],
        "best_test_accuracy": best_test,
        "deploy_validation_accuracy": deploy_validation,
        "deploy_test_accuracy": deploy_test,
        "best_iteration": best_iteration,
        "deploy_iteration": deploy_iteration,
        "promoted": promoted,
        "route_best_model": str(route_best_path),
        "route_deploy_model": str(route_deploy_path),
    }

def _collect_samples(
    strategy: SklearnIncrementalStrategy,
    routes: tuple[str, ...],
    episodes: int,
    frames: int,
    episode_offset: int,
) -> tuple[list[list[float]], list[int]]:
    # labels still come from a*, only the sampled states come from routes
    features: list[list[float]] = []
    labels: list[int] = []
    teacher = AStarStrategy()
    for route_name in routes:
        for episode in range(episodes):
            simulation = GameSimulation(
                strategy=AStarStrategy(),
                telemetry_config=TelemetryConfig(enabled=False),
                seed=7 + episode + episode_offset,
            )
            try:
                for frame in range(frames):
                    metrics = simulation.step(_route_input(route_name, frame, episode + episode_offset), 1.0 / 60.0)
                    if metrics.target is None or not metrics.decision.cells:
                        continue
                    start = metrics.decision.cells[0]
                    goal = simulation.world.nearest_walkable_cell(simulation.world.world_to_cell(metrics.target))
                    if goal is None:
                        continue
                    path = teacher._find_path(simulation.world, start, goal)
                    if len(path) < 2:
                        continue
                    move = (path[1][0] - start[0], path[1][1] - start[1])
                    if move not in MOVE_LABELS:
                        continue
                    features.append(strategy._features(simulation.world, start, goal))
                    labels.append(MOVE_LABELS.index(move))
            finally:
                simulation.close()
    return features, labels

def _load_or_create_model(strategy: SklearnIncrementalStrategy, world: WorldState) -> object:
    # start from the current best model when it matches this feature set
    model_path = ROOT / strategy.best_model_path
    if not model_path.exists():
        model_path = ROOT / strategy.model_path
    if model_path.exists():
        model = load(model_path)
        if strategy._model_matches_settings(model) and strategy._model_matches_features(model, strategy._feature_count(world)):
            return model
    return strategy._new_classifier(SGDClassifier)

def _score_sets(
    model: object,
    train_x: list[list[float]],
    train_y: list[int],
    validation_x: list[list[float]],
    validation_y: list[int],
    test_x: list[list[float]],
    test_y: list[int],
    random_x: list[list[float]],
    random_y: list[int],
) -> dict[str, float]:
    return {
        "train_accuracy": _safe_score(model, train_x, train_y),
        "validation_accuracy": _safe_score(model, validation_x, validation_y),
        "test_accuracy": _safe_score(model, test_x, test_y),
        "random_accuracy": _safe_score(model, random_x, random_y),
    }

def _safe_score(model: object, features: list[list[float]], labels: list[int]) -> float:
    try:
        return float(model.score(features, labels))  # type: ignore[attr-defined]
    except NotFittedError:
        return 0.0

def _log_row(
    iteration: int,
    train_routes: tuple[str, ...],
    test_routes: tuple[str, ...],
    train_x: list[list[float]],
    validation_x: list[list[float]],
    test_x: list[list[float]],
    random_x: list[list[float]],
    scores: dict[str, float],
    baseline: dict[str, float],
    saved_best: bool,
    saved_deployable: bool,
    promoted: bool,
    route_model_path: Path,
    route_best_path: Path,
    route_deploy_path: Path,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "train_routes": ",".join(train_routes),
        "validation_routes": ",".join(train_routes),
        "test_routes": ",".join(test_routes),
        "train_samples": len(train_x),
        "validation_samples": len(validation_x),
        "test_samples": len(test_x),
        "random_eval_samples": len(random_x),
        "train_accuracy": f"{scores['train_accuracy']:.4f}",
        "validation_accuracy": f"{scores['validation_accuracy']:.4f}",
        "test_accuracy": f"{scores['test_accuracy']:.4f}",
        "random_accuracy": f"{scores['random_accuracy']:.4f}",
        "validation_delta": f"{scores['validation_accuracy'] - baseline['validation_accuracy']:.4f}",
        "test_delta": f"{scores['test_accuracy'] - baseline['test_accuracy']:.4f}",
        "saved_route_best": int(saved_best),
        "saved_deployable": int(saved_deployable),
        "promoted_active": int(promoted),
        "route_model_path": str(route_model_path),
        "route_best_model_path": str(route_best_path),
        "route_deploy_model_path": str(route_deploy_path),
    }

def _shuffled(features: list[list[float]], labels: list[int], seed: int) -> tuple[list[list[float]], list[int]]:
    pairs = list(zip(features, labels))
    random.Random(seed).shuffle(pairs)
    shuffled_x, shuffled_y = zip(*pairs)
    return list(shuffled_x), list(shuffled_y)

def _write_route_metadata(
    path: Path,
    scores: dict[str, float],
    train_routes: tuple[str, ...],
    test_routes: tuple[str, ...],
    iteration: int,
    model_path: Path,
) -> None:
    data = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "iteration": iteration,
        "train_routes": list(train_routes),
        "test_routes": list(test_routes),
        "feature_set": FEATURE_SET_NAME,
        "eval_accuracy": scores["validation_accuracy"],
        "train_accuracy": scores["train_accuracy"],
        "validation_accuracy": scores["validation_accuracy"],
        "test_accuracy": scores["test_accuracy"],
        "random_accuracy": scores["random_accuracy"],
        "model_path": str(model_path),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def _promote_route_model(
    strategy: SklearnIncrementalStrategy,
    route_best_path: Path,
    route_metadata_path: Path,
) -> bool:
    if not route_best_path.exists():
        return False
    model = load(route_best_path)
    dump(model, ROOT / strategy.model_path)
    dump(model, ROOT / strategy.best_model_path)
    if route_metadata_path.exists():
        data = json.loads(route_metadata_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data["promoted_to_active"] = True
    data["active_model_path"] = str(ROOT / strategy.best_model_path)
    data["saved_at"] = datetime.now().isoformat(timespec="seconds")
    (ROOT / strategy.best_metadata_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True

def _is_deployable(scores: dict[str, float], baseline: dict[str, float]) -> bool:
    # active model should improve route val without losing broad map skill
    return (
        scores["validation_accuracy"] > baseline["validation_accuracy"]
        and scores["test_accuracy"] >= baseline["test_accuracy"] * 0.98
        and scores["random_accuracy"] >= baseline["random_accuracy"] * 0.95
    )

def _mark_promoted(log_path: Path, iteration: int) -> None:
    rows = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
    if not rows:
        return
    for row in rows:
        if row["iteration"] == str(iteration):
            row["promoted_active"] = "1"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def _ensure_samples(*sample_sets: list[list[float]]) -> None:
    if any(len(samples) == 0 for samples in sample_sets):
        raise RuntimeError("Route training needs non-empty train, validation, test, and random samples.")

def _route_training_log_path() -> Path:
    directory = ROOT / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return directory / f"route_ml_training_{stamp}.csv"

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ML policy on route-split teacher samples.")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--train-routes", default=",".join(TRAIN_ROUTES))
    parser.add_argument("--test-routes", default=",".join(TEST_ROUTES))
    parser.add_argument("--random-samples", type=int, default=512)
    args = parser.parse_args()
    summary = run_route_training(
        episodes=args.episodes,
        frames=args.frames,
        passes=args.passes,
        train_routes=_split_routes(args.train_routes),
        test_routes=_split_routes(args.test_routes),
        random_samples=args.random_samples,
    )
    print(f"Route ML training complete: log={summary['log']}")
    print(
        "Validation accuracy: "
        f"{summary['baseline_validation_accuracy']:.4f} -> {summary['best_validation_accuracy']:.4f}"
    )
    print(f"Test accuracy: {summary['baseline_test_accuracy']:.4f} -> {summary['best_test_accuracy']:.4f}")
    print(f"Deploy iteration: {summary['deploy_iteration']}")
    print(f"Promoted active model: {int(bool(summary['promoted']))}")

def _split_routes(value: str) -> tuple[str, ...]:
    return tuple(route.strip() for route in value.split(",") if route.strip())

if __name__ == "__main__":
    main()
