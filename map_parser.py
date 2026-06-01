"""
map_parser.py
=============
Loads occupancy grids from map.png files.

Key improvement over previous version:
  - load_grid_auto() works WITHOUT metrics.json by estimating the cell
    grid size directly from the image (counts repeating obstacle columns/rows).
  - load_grid_cell_resolution() still works with metrics.json when available.
  - Both functions handle thin-wall mazes via centre-pixel sampling.
"""

import cv2
import numpy as np
import json
import os
from typing import Tuple, Optional


def load_occupancy_grid(map_path: str) -> np.ndarray:
    """
    Load map.png and return a binary obstacle grid at pixel resolution.
    Returns uint8 array: 1=wall, 0=free.
    """
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open: {map_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 1, cv2.THRESH_BINARY_INV)
    binary = binary[3:-3, 3:-3]
    return binary.astype(np.uint8)


def load_grid_cell_resolution(map_path: str,
                               metrics_path: str
                               ) -> Tuple[np.ndarray, int, int]:
    """
    Load map at logical cell resolution using metrics.json for dimensions.
    Samples centre pixel of each cell — critical for thin-wall mazes.
    Returns (grid, cell_w, cell_h).
    """
    with open(metrics_path) as f:
        m = json.load(f)
    cell_w = m["width"]
    cell_h = m["height"]
    binary = load_occupancy_grid(map_path)
    return _sample_cell_centres(binary, cell_w, cell_h), cell_w, cell_h


def load_grid_auto(map_path: str,
                   metrics_path: Optional[str] = None
                   ) -> Tuple[np.ndarray, int, int]:
    """
    Universal loader — works with OR without metrics.json.

    If metrics.json exists it is used for exact dimensions.
    Otherwise cell size is estimated from image periodicity.
    Returns (grid, cell_w, cell_h).
    """
    if metrics_path and os.path.exists(metrics_path):
        return load_grid_cell_resolution(map_path, metrics_path)

    binary = load_occupancy_grid(map_path)
    px_h, px_w = binary.shape
    cell_w, cell_h = _estimate_cell_size(binary)

    if cell_w > 0 and cell_h > 0:
        return _sample_cell_centres(binary, cell_w, cell_h), cell_w, cell_h
    return binary, px_w, px_h


def _sample_cell_centres(binary: np.ndarray,
                          cell_w: int, cell_h: int) -> np.ndarray:
    px_h, px_w = binary.shape
    result = np.zeros((cell_h, cell_w), dtype=np.uint8)
    for gy in range(cell_h):
        for gx in range(cell_w):
            px = min(int((gx + 0.5) * px_w / cell_w), px_w - 1)
            py = min(int((gy + 0.5) * px_h / cell_h), px_h - 1)
            result[gy, gx] = binary[py, px]
    return result


def _estimate_cell_size(binary: np.ndarray) -> Tuple[int, int]:
    """Estimate logical cell size via autocorrelation of column/row sums."""
    col_sums = binary.sum(axis=0).astype(np.float32)
    row_sums = binary.sum(axis=1).astype(np.float32)
    period_x = _dominant_period(col_sums)
    period_y = _dominant_period(row_sums)
    px_h, px_w = binary.shape
    if 2 <= period_x <= px_w // 2 and 2 <= period_y <= px_h // 2:
        return px_w // period_x, px_h // period_y
    return 0, 0


def _dominant_period(signal: np.ndarray) -> int:
    n = len(signal)
    if n < 4:
        return 0
    s = signal - signal.mean()
    corr = np.correlate(s, s, mode="full")[n - 1:]
    corr[0] = 0
    max_lag = min(n // 2, 60)
    return int(np.argmax(corr[1:max_lag])) + 1


def load_scenario(scenario_path: str) -> dict:
    with open(scenario_path) as f:
        return json.load(f)


def load_tasks(tasks_path: str) -> list:
    if not os.path.exists(tasks_path):
        return []
    with open(tasks_path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def grid_to_display(grid: np.ndarray) -> np.ndarray:
    display = (1 - grid) * 255
    return cv2.cvtColor(display.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def print_grid_info(grid: np.ndarray, scenario: dict) -> None:
    h, w  = grid.shape
    total = h * w
    obs   = int(np.sum(grid))
    print("=" * 42)
    print("  MAP INFO")
    print("=" * 42)
    print(f"  Grid size    : {w} x {h} cells")
    print(f"  Obstacles    : {obs}  ({obs/total*100:.1f}%)")
    print(f"  Free         : {total-obs}  ({(total-obs)/total*100:.1f}%)")
    print(f"  Scenario type: {scenario.get('type', 'unknown')}")
    print(f"  Agents       : {scenario.get('num_agents', '?')}")
    print("=" * 42)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    data_dir = "data"
    cases = sorted([d for d in os.listdir(data_dir)
                    if os.path.isdir(f"{data_dir}/{d}")]) if os.path.exists(data_dir) else []
    for case in cases:
        base = f"{data_dir}/{case}"
        map_path     = f"{base}/map.png"
        metrics_path = f"{base}/metrics.json"
        sc_path      = f"{base}/scenario.json"
        if not os.path.exists(map_path):
            continue
        print(f"\n── {case} ──")
        grid_a, gw, gh = load_grid_auto(map_path)
        print(f"  Auto   : {gw}x{gh}  density={grid_a.sum()/(gw*gh):.3f}")
        if os.path.exists(metrics_path):
            grid_m, gw_m, gh_m = load_grid_cell_resolution(map_path, metrics_path)
            print(f"  Metrics: {gw_m}x{gh_m}  density={grid_m.sum()/(gw_m*gh_m):.3f}")