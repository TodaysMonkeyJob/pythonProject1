import json
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import SVC, SVR

from core.bus import CONTROL_STOP, create_queue, receive
from core.events import Event
from modules.gas.data import (CLASS_NAMES, CLEAN_AIR, DEFAULT_VOLTAGES, GAS_SHEETS, demo_replay_curves, load_dataset,
                              parse_sheet, read_replay_csv, write_clean_workbook, write_replay_csv)
from modules.gas.inference import GasPredictor
from modules.gas.train import augment, run_training, split_indices
from modules.gas.worker import CsvReplayReader, GasWorker, TrendTracker

REPO = Path(__file__).resolve().parent.parent
DATA_PATH = REPO / "data" / "gas" / "data.xls"
REPLAY_PATH = REPO / "data" / "gas" / "replay_demo.csv"

# Hyper-parameters picked by the full grid search; fixed here to keep the test run fast
TEST_GRIDS = {"SVC": (SVC(probability=True, random_state=42),
                      {"model__C": [0.1], "model__kernel": ["poly"], "model__gamma": [100]})}
TEST_REGRESSOR_GRID = [{"model": [SVR()], "model__C": [100], "model__gamma": ["scale"], "model__epsilon": [0.01]}]

GAS_CONFIG = {
    "gas_poll_sec": 1.0,
    "sensor_id": "gas-test",
    "thresholds": {"ethanol": 5.0, "ammonia": 3.0, "acetone": 5.0, "toluol": 3.0},
    "rise_rate_threshold": 2.0,
    "trend_window_sec": 60,
    "alert_cooldown_sec": 30,
}


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(str(DATA_PATH))


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("gas")
    metrics = run_training(str(DATA_PATH), str(root / "models"), str(root / "reports"),
                           classifier_grids=TEST_GRIDS, regressor_grid=TEST_REGRESSOR_GRID)
    predictor = GasPredictor(str(root / "models" / "gas_classifier.joblib"),
                             str(root / "models" / "gas_concentration.joblib"))
    return root, metrics, predictor


def _drain(q):
    events = []
    while True:
        try:
            events.append(Event.from_json(q.get_nowait()))
        except queue.Empty:
            return events


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# --- data -------------------------------------------------------------------------------------------------------

def test_loader_shape(dataset):
    assert dataset.X.shape == (144, 16)
    assert dataset.voltages == DEFAULT_VOLTAGES
    assert np.bincount(dataset.y_gas).tolist() == [35, 36, 33, 36, 4]
    assert set(dataset.y_conc[dataset.y_gas == CLEAN_AIR]) == {0.0}
    assert (dataset.y_conc[dataset.y_gas != CLEAN_AIR] > 0).all()
    assert np.isfinite(dataset.X).all()


def test_parse_sheet_ignores_header_position():
    curve = [0.5 + 0.01 * i for i in range(16)]
    raw = pd.DataFrame([
        ["Ug, V", None, *DEFAULT_VOLTAGES],
        [2.0, None, *curve],
        ["C, %", None, *[None] * 16],
        [0.0, None, *curve],
        [None] * 18,
    ])
    X, conc, voltages = parse_sheet(raw)
    assert X.shape == (2, 16)
    assert conc.tolist() == [0.0, 2.0]
    assert voltages == DEFAULT_VOLTAGES


def test_clean_workbook_round_trip(dataset, tmp_path):
    path = tmp_path / "clean.xlsx"
    write_clean_workbook(dataset, str(path))
    clean = load_dataset(str(path))
    assert np.allclose(clean.X, dataset.X)
    assert (clean.y_gas == dataset.y_gas).all()
    # Legacy layout: last row is "C, %" + voltages, no empty column
    raw = pd.read_excel(path, sheet_name="ethanol", header=None)
    assert raw.iloc[-1, 0] == "C, %"
    assert raw.shape[1] == 17


def test_replay_csv_round_trip(dataset, tmp_path):
    curves = demo_replay_curves(dataset)
    path = tmp_path / "replay.csv"
    write_replay_csv(curves, dataset.voltages, str(path))
    assert np.allclose(read_replay_csv(str(path)), curves, atol=1e-5)


# --- no leakage -------------------------------------------------------------------------------------------------

def test_split_is_stratified_and_disjoint(dataset):
    train_idx, test_idx = split_indices(dataset)
    assert not set(train_idx) & set(test_idx)
    assert len(train_idx) + len(test_idx) == len(dataset)
    assert set(dataset.y_gas[test_idx]) == set(range(len(CLASS_NAMES)))


def test_augmentation_uses_training_rows_only(dataset):
    train_idx, test_idx = split_indices(dataset)
    aug = augment(dataset, train_idx)
    assert len(aug.y_gas) > 0
    assert set(aug.parents.ravel()) <= set(train_idx)
    assert not set(aug.parents.ravel()) & set(test_idx)
    # Every synthetic row really is the midpoint of its two training parents from the same sheet
    a, b = aug.parents[:, 0], aug.parents[:, 1]
    assert np.allclose(aug.X, (dataset.X[a] + dataset.X[b]) / 2)
    assert (dataset.source[a] == dataset.source[b]).all()


def test_test_set_contains_only_original_rows(dataset):
    _, test_idx = split_indices(dataset)
    aug = augment(dataset, split_indices(dataset)[0])
    test_rows = {tuple(row) for row in dataset.X[test_idx].round(6)}
    synthetic = {tuple(row) for row in aug.X.round(6)}
    assert not test_rows & synthetic


# --- training + inference ---------------------------------------------------------------------------------------

def test_training_writes_artifacts_and_metrics(trained):
    root, metrics, _ = trained
    assert (root / "models" / "gas_classifier.joblib").stat().st_size < 20e6
    assert (root / "models" / "gas_concentration.joblib").stat().st_size < 20e6
    saved = json.loads((root / "reports" / "gas_metrics.json").read_text())
    assert saved["classifier"]["test_f1_macro"] == metrics["classifier"]["test_f1_macro"]
    assert saved["data"]["test_rows"] == 29
    assert (root / "reports" / "gas_confusion_matrix.png").exists()


def test_predictor_on_held_out_rows(trained, dataset):
    _, _, predictor = trained
    _, test_idx = split_indices(dataset)
    correct = 0
    for i in test_idx:
        result = predictor.predict(dataset.X[i].tolist())
        assert set(result) == {"gas", "gas_confidence", "concentration_pct"}
        assert result["gas"] in CLASS_NAMES
        assert 0.0 <= result["gas_confidence"] <= 1.0
        assert result["concentration_pct"] >= 0.0
        correct += result["gas"] == CLASS_NAMES[dataset.y_gas[i]]
    assert correct / len(test_idx) >= 0.95


def test_predictor_validates_input(trained):
    _, _, predictor = trained
    with pytest.raises(ValueError, match="Expected 16"):
        predictor.predict([0.5] * 15)
    with pytest.raises(ValueError, match="NaN"):
        predictor.predict([0.5] * 15 + [float("nan")])


def test_inference_latency(trained, dataset):
    _, _, predictor = trained
    curve = dataset.X[50].tolist()
    predictor.predict(curve)
    start = time.perf_counter()
    for _ in range(20):
        predictor.predict(curve)
    assert (time.perf_counter() - start) / 20 < 0.05


def test_runtime_modules_do_not_import_matplotlib():
    code = ("import sys, modules.gas.inference, modules.gas.worker; "
            "sys.exit(1 if 'matplotlib' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


# --- worker -----------------------------------------------------------------------------------------------------

def test_trend_tracker():
    trend = TrendTracker(window_sec=60, stable_band=0.2)
    assert trend.add(0, 1.0) == (None, "stable")
    trend.add(10, 1.0)
    rate, label = trend.add(20, 1.0)
    assert label == "stable" and abs(rate) < 1e-9
    rate, label = trend.add(30, 2.0)
    assert label == "rising" and rate > 0
    for t in range(40, 200, 10):  # old points drop out of the window
        rate, label = trend.add(t, 0.0)
    assert len(trend.points) == 7 and label == "stable"


def test_replay_reader_loops(tmp_path):
    path = tmp_path / "r.csv"
    write_replay_csv([[1.0] * 16, [2.0] * 16], DEFAULT_VOLTAGES, str(path))
    reader = CsvReplayReader(str(path), loop=True)
    assert [reader()[0] for _ in range(3)] == [1.0, 2.0, 1.0]
    reader = CsvReplayReader(str(path), loop=False)
    assert [reader()[0] for _ in range(3)] == [1.0, 2.0, 2.0]


def test_worker_replay_publishes_readings_and_alerts(trained):
    _, _, predictor = trained
    clock = FakeClock()
    config = {"gas": {**GAS_CONFIG, "mode": "replay", "replay_csv": str(REPLAY_PATH), "replay_loop": False}}
    worker = GasWorker(queue.Queue(), queue.Queue(), config, predictor=predictor, clock=clock)
    worker.setup()

    payloads = []
    for _ in range(len(read_replay_csv(str(REPLAY_PATH)))):
        payloads.append(worker.poll_once())
        clock.now += 1.0
    events = _drain(worker.event_queue)

    readings = [e for e in events if e.type == "gas_reading"]
    alerts = [e for e in events if e.type == "alert"]
    assert len(readings) == len(payloads)
    assert all(e.source == "gas" and e.severity == "info" for e in readings)
    assert readings[0].payload["gas"] == "clean_air"
    assert readings[-1].payload["gas"] == "acetone"
    assert readings[-1].payload["sensor_id"] == "gas-test"
    assert set(readings[-1].payload["curve"]) == {"r_min", "r_max", "r_mean", "peak_ug_v"}

    reasons = {a.payload["reason"] for a in alerts}
    assert {"gas_threshold", "gas_rising"} <= reasons
    assert all(a.severity == "critical" for a in alerts)
    first_threshold = next(a for a in alerts if a.payload["reason"] == "gas_threshold")
    assert first_threshold.payload["gas"] == "acetone"
    assert first_threshold.payload["concentration_pct"] >= 5.0
    # The reading that triggered the first threshold alert is published right before it; the one before
    # that was still below the threshold
    readings_before = [e for e in events[:events.index(first_threshold)] if e.type == "gas_reading"]
    assert readings_before[-1].payload["concentration_pct"] >= 5.0
    assert readings_before[-2].payload["concentration_pct"] < 5.0


class StubPredictor:
    voltages = list(DEFAULT_VOLTAGES)

    def __init__(self, concentrations):
        self.concentrations = iter(concentrations)

    def predict(self, curve):
        return {"gas": "ammonia", "gas_confidence": 0.9, "concentration_pct": next(self.concentrations)}


def test_worker_alert_cooldown():
    clock = FakeClock()
    worker = GasWorker(queue.Queue(), queue.Queue(), {"gas": GAS_CONFIG},
                       predictor=StubPredictor([4.0] * 40), reader=lambda: [0.5] * 16, clock=clock)
    for _ in range(40):
        worker.poll_once()
        clock.now += 1.0
    alerts = [e for e in _drain(worker.event_queue) if e.type == "alert"]
    assert [a.payload["reason"] for a in alerts] == ["gas_threshold", "gas_threshold"]  # t=0 and t=30


def test_worker_sensor_error_becomes_warning():
    def broken():
        raise OSError("i2c timeout")

    worker = GasWorker(queue.Queue(), queue.Queue(), {"gas": GAS_CONFIG},
                       predictor=StubPredictor([]), reader=broken)
    assert worker.poll_once() is None
    (event,) = _drain(worker.event_queue)
    assert event.type == "alert" and event.severity == "warning"
    assert event.payload["reason"] == "sensor_error"


def test_worker_run_loop_heartbeat_and_stop():
    event_queue, control_queue = create_queue(), create_queue()
    config = {"gas": {**GAS_CONFIG, "gas_poll_sec": 0.05}}
    worker = GasWorker(event_queue, control_queue, config,
                       predictor=StubPredictor([0.0] * 1000), reader=lambda: [0.5] * 16)
    thread = threading.Thread(target=worker.run)
    thread.start()
    received = [receive(event_queue, timeout=5) for _ in range(3)]
    control_queue.put(CONTROL_STOP)
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert (received[0].source, received[0].type) == ("system", "heartbeat")
    assert any(e.type == "gas_reading" for e in received)


def test_package_entry_point_is_lightweight():
    code = "import sys, modules.gas; sys.exit(1 if 'sklearn' in sys.modules else 0)"
    assert subprocess.run([sys.executable, "-c", code], cwd=REPO).returncode == 0


def test_gas_sheets_match_classes():
    assert CLASS_NAMES[:len(GAS_SHEETS)] == GAS_SHEETS
