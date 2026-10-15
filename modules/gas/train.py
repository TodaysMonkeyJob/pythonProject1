"""
Trains the gas-type classifier and per-gas concentration regressors.

Pipeline: load original rows → stratified train/test split → augment the training rows only
(midpoints between neighbouring concentrations of the same gas) → GridSearchCV over scaled models →
best classifier by macro-F1 on the untouched test set → concentration regressor per gas →
``models/gas_classifier.joblib``, ``models/gas_concentration.joblib``, ``reports/gas_metrics.json``,
``reports/gas_confusion_matrix.png``.

Run from the repo root: ``python -m modules.gas.train`` (``--quick`` uses small grids for a smoke run).
"""
import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier

from modules.gas.data import CLASS_NAMES, CLEAN_AIR, DEFAULT_DATA_PATH, GAS_SHEETS, GasDataset, load_dataset

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
DEFAULT_MODELS_DIR = "models"
DEFAULT_REPORTS_DIR = "reports"
CLASSIFIER_FILE = "gas_classifier.joblib"
REGRESSOR_FILE = "gas_concentration.joblib"

# Same grids as the legacy script, prefixed for the Pipeline step "model"
CLASSIFIER_GRIDS = {
    "SVC": (SVC(probability=True, random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10], "model__kernel": ["linear", "rbf", "poly"],
             "model__gamma": [0.1, 1, 10, 100]}),
    "GradientBoostingClassifier": (GradientBoostingClassifier(random_state=RANDOM_STATE),
                                   {"model__n_estimators": [50, 100, 150, 200],
                                    "model__learning_rate": [0.01, 0.05, 0.1, 0.2]}),
    "KNeighborsClassifier": (KNeighborsClassifier(), {"model__n_neighbors": [5, 10, 15, 20]}),
    "DecisionTreeClassifier": (DecisionTreeClassifier(random_state=RANDOM_STATE),
                               {"model__max_depth": [None, 10, 20, 30]}),
}

QUICK_CLASSIFIER_GRIDS = {
    "SVC": (SVC(probability=True, random_state=RANDOM_STATE), {"model__C": [10], "model__kernel": ["rbf"],
                                                                "model__gamma": [0.1]}),
    "DecisionTreeClassifier": (DecisionTreeClassifier(random_state=RANDOM_STATE), {"model__max_depth": [None]}),
}

REGRESSOR_GRID = [
    {"model": [SVR()], "model__C": [1, 10, 100], "model__gamma": ["scale", 0.1, 1], "model__epsilon": [0.01, 0.1]},
    {"model": [GradientBoostingRegressor(random_state=RANDOM_STATE)],
     "model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
]

QUICK_REGRESSOR_GRID = [{"model": [SVR()], "model__C": [100], "model__gamma": ["scale"], "model__epsilon": [0.01]}]


@dataclass
class Augmented:
    """
    Synthetic rows built from training rows only. parents[i] = indices (into the dataset) of the two
    original rows whose midpoint is row i.
    """
    X: np.ndarray
    y_gas: np.ndarray
    y_conc: np.ndarray
    source: np.ndarray
    parents: np.ndarray


def split_indices(dataset: GasDataset, test_size: float = TEST_SIZE,
                  random_state: int = RANDOM_STATE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split of the ORIGINAL rows, done before any augmentation.

    :return: (train_idx, test_idx) — indices into the dataset
    """
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, stratify=dataset.y_gas,
                                           random_state=random_state)
    return np.sort(train_idx), np.sort(test_idx)


def augment(dataset: GasDataset, train_idx: Sequence[int]) -> Augmented:
    """
    Adds a midpoint between every two neighbouring (by concentration) training rows of the same sheet,
    like the legacy calculate_average_sum(), but using training rows only.

    :param dataset: Full dataset
    :param train_idx: Training row indices
    :return: Augmented rows
    """
    train_idx = np.asarray(train_idx)
    X, y_gas, y_conc, source, parents = [], [], [], [], []
    for sheet in range(len(GAS_SHEETS)):
        rows = train_idx[dataset.source[train_idx] == sheet]
        rows = rows[np.argsort(dataset.y_conc[rows], kind="stable")]
        for a, b in zip(rows[:-1], rows[1:], strict=True):
            conc = (dataset.y_conc[a] + dataset.y_conc[b]) / 2
            X.append((dataset.X[a] + dataset.X[b]) / 2)
            y_gas.append(CLEAN_AIR if conc == 0 else sheet)
            y_conc.append(conc)
            source.append(sheet)
            parents.append((a, b))
    return Augmented(X=np.array(X).reshape(-1, dataset.X.shape[1]), y_gas=np.array(y_gas, dtype=int),
                     y_conc=np.array(y_conc), source=np.array(source, dtype=int),
                     parents=np.array(parents, dtype=int).reshape(-1, 2))


def training_set(dataset: GasDataset, train_idx: Sequence[int],
                 aug: Augmented) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    :return: (X, y_gas, y_conc, source) = original training rows + augmented rows
    """
    return (np.vstack([dataset.X[train_idx], aug.X]),
            np.concatenate([dataset.y_gas[train_idx], aug.y_gas]),
            np.concatenate([dataset.y_conc[train_idx], aug.y_conc]),
            np.concatenate([dataset.source[train_idx], aug.source]))


def make_pipeline(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def train_classifiers(X_train, y_train, X_test, y_test, grids=None) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Grid-searches every classifier and evaluates it on the test set.

    :return: (best model name, {name: fitted pipeline}, {name: metrics})
    """
    grids = grids or CLASSIFIER_GRIDS
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    labels = list(range(len(CLASS_NAMES)))
    fitted, results = {}, {}
    for name, (model, grid) in grids.items():
        search = GridSearchCV(make_pipeline(model), grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)
        y_pred = search.best_estimator_.predict(X_test)
        fitted[name] = search.best_estimator_
        results[name] = {
            "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items()},
            "cv_f1_macro": round(float(search.best_score_), 4),
            "test_f1_macro": round(float(f1_score(y_test, y_pred, labels=labels, average="macro",
                                                  zero_division=0)), 4),
            "test_accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        }
        print(f"== {name}  {results[name]}")
        print(classification_report(y_test, y_pred, labels=labels, target_names=CLASS_NAMES, zero_division=0))
    best = max(results, key=lambda n: (results[n]["test_f1_macro"], results[n]["cv_f1_macro"]))
    return best, fitted, results


def train_regressors(X_train, y_conc_train, source_train, gas_train,
                     X_test, y_conc_test, gas_test, grid=None) -> Tuple[Dict[str, Pipeline], Dict[str, Any]]:
    """
    One concentration regressor per gas, trained on that gas's sheet (including its clean-air rows).

    :return: ({gas: fitted pipeline}, {gas: metrics})
    """
    grid = grid or REGRESSOR_GRID
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    models, results = {}, {}
    for sheet, gas in enumerate(GAS_SHEETS):
        mask = source_train == sheet
        search = GridSearchCV(make_pipeline(SVR()), grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
        search.fit(X_train[mask], y_conc_train[mask])
        test_mask = gas_test == sheet
        y_pred = np.clip(search.best_estimator_.predict(X_test[test_mask]), 0, None)
        models[gas] = search.best_estimator_
        results[gas] = {
            "model": type(search.best_params_["model"]).__name__,
            "best_params": {k.removeprefix("model__"): v for k, v in search.best_params_.items() if k != "model"},
            "cv_mae_pct": round(float(-search.best_score_), 4),
            "test_mae_pct": round(float(mean_absolute_error(y_conc_test[test_mask], y_pred)), 4),
            "n_test": int(test_mask.sum()),
            "conc_range_pct": [float(y_conc_train[mask].min()), float(y_conc_train[mask].max())],
        }
        print(f"== regressor {gas}: {results[gas]}")
    return models, results


def save_confusion_matrix(matrix: np.ndarray, path: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def repeated_split_scores(dataset: GasDataset, estimator, n_splits: int = 10) -> Dict[str, Any]:
    """
    Re-fits the selected estimator (fixed hyper-parameters) on several stratified splits, augmenting the
    training part only, to show how much the single-split test score depends on the split.

    :return: Summary of test macro-F1 over the splits
    """
    labels = list(range(len(CLASS_NAMES)))
    scores = []
    for seed in range(n_splits):
        train_idx, test_idx = split_indices(dataset, random_state=seed)
        X, y, _, _ = training_set(dataset, train_idx, augment(dataset, train_idx))
        model = clone(estimator).fit(X, y)
        y_pred = model.predict(dataset.X[test_idx])
        scores.append(f1_score(dataset.y_gas[test_idx], y_pred, labels=labels, average="macro", zero_division=0))
    return {"n_splits": n_splits, "test_f1_macro_mean": round(float(np.mean(scores)), 4),
            "test_f1_macro_std": round(float(np.std(scores)), 4), "test_f1_macro_min": round(float(np.min(scores)), 4)}


def _latency_ms(predictor, curves: np.ndarray, repeats: int = 5) -> Dict[str, float]:
    timings = []
    for _ in range(repeats):
        for curve in curves:
            start = time.perf_counter()
            predictor.predict(curve.tolist())
            timings.append((time.perf_counter() - start) * 1000)
    return {"mean": round(float(np.mean(timings)), 3), "p95": round(float(np.percentile(timings, 95)), 3)}


def run_training(data_path: str = DEFAULT_DATA_PATH, models_dir: str = DEFAULT_MODELS_DIR,
                 reports_dir: str = DEFAULT_REPORTS_DIR, quick: bool = False,
                 classifier_grids=None, regressor_grid=None) -> Dict[str, Any]:
    """
    Full training run. Writes both models and the reports.

    :param quick: Use the small grids (smoke run)
    :param classifier_grids: Overrides the classifier grids (tests)
    :param regressor_grid: Overrides the regressor grid (tests)

    :return: Metrics dict (also saved to reports/gas_metrics.json)
    """
    from modules.gas.inference import GasPredictor

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    dataset = load_dataset(data_path)
    train_idx, test_idx = split_indices(dataset)
    aug = augment(dataset, train_idx)
    X_train, y_gas_train, y_conc_train, source_train = training_set(dataset, train_idx, aug)
    X_test, y_gas_test, y_conc_test = dataset.X[test_idx], dataset.y_gas[test_idx], dataset.y_conc[test_idx]
    print(f"Original rows: {len(dataset)} (train {len(train_idx)}, test {len(test_idx)}), "
          f"augmented train rows: {len(aug.y_gas)}")

    with warnings.catch_warnings():
        # clean_air has only 3 training rows: some CV folds miss it
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        best_name, classifiers, clf_results = train_classifiers(
            X_train, y_gas_train, X_test, y_gas_test,
            classifier_grids or (QUICK_CLASSIFIER_GRIDS if quick else None))
        regressors, reg_results = train_regressors(
            X_train, y_conc_train, source_train, y_gas_train, X_test, y_conc_test, y_gas_test,
            regressor_grid or (QUICK_REGRESSOR_GRID if quick else None))

        best = classifiers[best_name]
        robustness = repeated_split_scores(dataset, best, n_splits=3 if quick else 10)

    y_pred = best.predict(X_test)
    labels = list(range(len(CLASS_NAMES)))
    matrix = confusion_matrix(y_gas_test, y_pred, labels=labels)
    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    common = {"voltages": list(dataset.voltages), "class_names": list(CLASS_NAMES),
              "sklearn_version": sklearn.__version__, "trained_at": trained_at}

    classifier_path = os.path.join(models_dir, CLASSIFIER_FILE)
    regressor_path = os.path.join(models_dir, REGRESSOR_FILE)
    joblib.dump({**common, "model": best, "model_name": best_name, "metrics": clf_results[best_name]},
                classifier_path, compress=3)
    joblib.dump({**common, "models": regressors, "metrics": reg_results}, regressor_path, compress=3)

    predictor = GasPredictor(classifier_path, regressor_path)
    end_to_end = [predictor.predict(curve.tolist()) for curve in X_test]
    e2e_conc = np.array([p["concentration_pct"] for p in end_to_end])

    metrics = {
        **common,
        "data": {
            "path": data_path,
            "original_rows": int(len(dataset)),
            "train_rows": int(len(train_idx)),
            "test_rows": int(len(test_idx)),
            "augmented_train_rows": int(len(aug.y_gas)),
            "class_counts": {name: int((dataset.y_gas == i).sum()) for i, name in enumerate(CLASS_NAMES)},
            "test_set": "original rows only, split before augmentation (stratified, random_state=42)",
        },
        "classifier": {
            "selected": best_name,
            "selection": "best macro-F1 on the test set (optimistic for the selected model)",
            "test_f1_macro": clf_results[best_name]["test_f1_macro"],
            "test_accuracy": clf_results[best_name]["test_accuracy"],
            "repeated_splits": robustness,
            "report": classification_report(y_gas_test, y_pred, labels=labels, target_names=CLASS_NAMES,
                                            zero_division=0, output_dict=True),
            "confusion_matrix": matrix.tolist(),
            "all_models": clf_results,
        },
        "concentration": {
            "per_gas": reg_results,
            "end_to_end_test_mae_pct": round(float(mean_absolute_error(y_conc_test, e2e_conc)), 4),
        },
        "runtime": {
            "inference_ms": _latency_ms(predictor, X_test),
            "model_size_mb": {
                CLASSIFIER_FILE: round(os.path.getsize(classifier_path) / 1e6, 3),
                REGRESSOR_FILE: round(os.path.getsize(regressor_path) / 1e6, 3),
            },
            "note": "measured on the training machine; re-check on the Raspberry Pi",
        },
    }
    with open(os.path.join(reports_dir, "gas_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    save_confusion_matrix(matrix, os.path.join(reports_dir, "gas_confusion_matrix.png"),
                          f"{best_name}, test macro-F1 = {clf_results[best_name]['test_f1_macro']:.3f}")

    print(f"Selected {best_name}: test macro-F1 {clf_results[best_name]['test_f1_macro']}, "
          f"end-to-end concentration MAE {metrics['concentration']['end_to_end_test_mae_pct']} %, "
          f"inference {metrics['runtime']['inference_ms']} ms")
    return metrics


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train the gas classifier and concentration regressors")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--reports-dir", default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--quick", action="store_true", help="Small grids for a fast smoke run")
    args = parser.parse_args(argv)
    run_training(args.data, args.models_dir, args.reports_dir, args.quick)


if __name__ == "__main__":
    main()
