"""
Gas sensor dataset loader.

Each sheet of ``data.xls`` holds one gas. A data row is one sensor response curve: concentration C (%)
in column 0, then normalized resistance R at 16 gate voltages Ug = 0.0..1.5 V. Header rows (``C, %``,
``Ug, V`` + voltages) and empty columns may sit anywhere, so the loader detects them instead of relying
on fixed positions. Rows with C = 0 (clean air) look the same for every gas and become the ``clean_air`` class.

``python -m modules.gas.data`` writes ``data/gas/data_clean.xlsx`` — the same data in the layout the legacy
research script expects (data rows, then a last row ``C, %`` + voltages, no empty column).
"""
import argparse
import csv
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

GAS_SHEETS = ("ethanol", "ammonia", "acetone", "toluol")
CLASS_NAMES = GAS_SHEETS + ("clean_air",)
CLEAN_AIR = CLASS_NAMES.index("clean_air")
N_FEATURES = 16
DEFAULT_VOLTAGES = tuple(round(0.1 * i, 1) for i in range(N_FEATURES))
DEFAULT_DATA_PATH = "data/gas/data.xls"
DEFAULT_CLEAN_PATH = "data/gas/data_clean.xlsx"
DEFAULT_REPLAY_PATH = "data/gas/replay_demo.csv"


@dataclass
class GasDataset:
    """
    All original rows of the workbook.

    X: (n, 16) resistance curves; y_gas: class index into CLASS_NAMES; y_conc: concentration, %;
    source: index of the sheet (gas) the row came from — also for clean-air rows.
    """
    X: np.ndarray
    y_gas: np.ndarray
    y_conc: np.ndarray
    source: np.ndarray
    voltages: Tuple[float, ...]

    def __len__(self) -> int:
        return len(self.y_gas)


def _to_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(number) else number


def parse_sheet(raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Tuple[float, ...]]:
    """
    Extracts curves from one raw sheet (read with header=None).

    :param raw: Sheet as read by pandas without a header
    :return: (X (n, 16), concentrations (n,), voltages (16,)), rows sorted by concentration
    """
    numeric = raw.map(_to_float)
    is_data_row = numeric.iloc[:, 0].notna() & (numeric.iloc[:, 1:].notna().sum(axis=1) >= N_FEATURES)
    data = numeric[is_data_row]
    headers = numeric[~is_data_row]
    if data.empty:
        raise ValueError("No data rows found")

    feature_cols = [c for c in data.columns[1:] if data[c].notna().all()]
    if len(feature_cols) != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} feature columns, found {len(feature_cols)}")

    # Voltage row: the header row with a number in every feature column
    voltage_rows = headers[headers[feature_cols].notna().all(axis=1)]
    if len(voltage_rows):
        voltages = tuple(round(v, 3) for v in voltage_rows.iloc[-1][feature_cols])
    else:
        voltages = DEFAULT_VOLTAGES

    order = np.argsort(voltages)
    voltages = tuple(voltages[i] for i in order)
    X = data[feature_cols].to_numpy(dtype=float)[:, order]
    conc = data.iloc[:, 0].to_numpy(dtype=float)
    by_conc = np.argsort(conc, kind="stable")
    return X[by_conc], conc[by_conc], voltages


def load_dataset(path: str = DEFAULT_DATA_PATH) -> GasDataset:
    """
    :param path: Workbook with one sheet per gas (raw data.xls or data_clean.xlsx)
    :return: GasDataset with every original row
    """
    sheets = pd.read_excel(path, sheet_name=list(GAS_SHEETS), header=None)
    parts_X, parts_gas, parts_conc, parts_src = [], [], [], []
    voltages = None
    for source, gas in enumerate(GAS_SHEETS):
        X, conc, sheet_voltages = parse_sheet(sheets[gas])
        if voltages is None:
            voltages = sheet_voltages
        elif not np.allclose(voltages, sheet_voltages):
            raise ValueError(f"Sheet {gas!r} uses different voltages: {sheet_voltages}")
        parts_X.append(X)
        parts_gas.append(np.where(conc == 0, CLEAN_AIR, source))
        parts_conc.append(conc)
        parts_src.append(np.full(len(conc), source))
    return GasDataset(X=np.vstack(parts_X),
                      y_gas=np.concatenate(parts_gas).astype(int),
                      y_conc=np.concatenate(parts_conc),
                      source=np.concatenate(parts_src).astype(int),
                      voltages=voltages)


def write_clean_workbook(dataset: GasDataset, path: str = DEFAULT_CLEAN_PATH) -> None:
    """
    Writes the dataset in the legacy layout: per sheet, rows "C, X1..X16" and a last row "C, %" + voltages.

    :param dataset: Loaded dataset
    :param path: Output .xlsx path
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for source, gas in enumerate(GAS_SHEETS):
            mask = dataset.source == source
            rows = np.column_stack([dataset.y_conc[mask], dataset.X[mask]]).tolist()
            rows.append(["C, %", *dataset.voltages])
            pd.DataFrame(rows).to_excel(writer, sheet_name=gas, header=False, index=False)


def write_replay_csv(curves: Sequence[Sequence[float]], voltages: Sequence[float], path: str) -> None:
    """
    Writes curves for the worker's replay mode: header = voltages, one curve per row.

    :param curves: Iterable of 16-value curves
    :param voltages: Feature voltages for the header
    :param path: Output CSV path
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"{v:.1f}" for v in voltages])
        for curve in curves:
            writer.writerow([f"{value:.5f}" for value in curve])


def read_replay_csv(path: str) -> List[List[float]]:
    """
    :param path: CSV written by write_replay_csv()
    :return: List of curves
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    curves = [[float(value) for value in row] for row in rows[1:] if row]
    if not curves:
        raise ValueError(f"No curves in {path}")
    return curves


def demo_replay_curves(dataset: GasDataset, gas: str = "acetone", clean_air_rows: int = 10,
                       hold_rows: int = 10) -> List[np.ndarray]:
    """
    Builds a demo scenario: clean air, then the gas rising one concentration step per row, then holding.

    :return: List of curves
    """
    clean = dataset.X[dataset.y_gas == CLEAN_AIR][0]
    mask = dataset.y_gas == GAS_SHEETS.index(gas)
    rising = dataset.X[mask][np.argsort(dataset.y_conc[mask])]
    return [clean] * clean_air_rows + list(rising) + [rising[-1]] * hold_rows


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Convert the raw gas workbook into the clean layout "
                                                 "and write the demo replay CSV")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH)
    parser.add_argument("--out", default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--replay-out", default=DEFAULT_REPLAY_PATH)
    args = parser.parse_args(argv)
    dataset = load_dataset(args.data)
    write_clean_workbook(dataset, args.out)
    print(f"{len(dataset)} rows from {args.data} -> {args.out}")
    curves = demo_replay_curves(dataset)
    write_replay_csv(curves, dataset.voltages, args.replay_out)
    print(f"{len(curves)} demo curves -> {args.replay_out}")


if __name__ == "__main__":
    main()
