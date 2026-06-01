"""
cv_evaluator.py
===============
Evaluates the CV pipeline quality across all available cases.

Metrics produced:
  1. Map parsing quality
       - Obstacle density error vs. metrics.json ground truth
       - Connectivity ratio (all free cells reachable from one source)
       - Parse time (ms)

  2. Agent detection accuracy
       - Per-agent grid error (Manhattan distance, CV coords vs. JSON ground truth)
       - Detection rate (fraction of agents found visually)
       - Detection time (ms)

  3. P/D marker detection (MAPD cases only)
       - Per-marker grid error
       - Detection rate

  4. Pipeline timing breakdown
       - Stage-by-stage table for every case

Results are printed as formatted tables AND saved to output/cv_evaluation.json.

Usage:
    python3 cv_evaluator.py              # all cases in data/
    python3 cv_evaluator.py --case 1     # specific case (1-based index)
    python3 cv_evaluator.py --verbose    # show per-agent detail
"""

import os
import sys
import json
import time
import argparse
from collections import deque
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np

sys.path.insert(0, ".")
from map_parser     import load_grid_cell_resolution, load_grid_auto, load_scenario, load_tasks
from agent_detector import detect_agents, detect_markers


# ── connectivity check ────────────────────────────────────────────────────────

def connectivity_ratio(grid: np.ndarray) -> float:
    """
    Fraction of free cells reachable from the first free cell (BFS).
    1.0 means the map is fully connected.
    """
    gh, gw = grid.shape
    free_cells = [(x, y) for y in range(gh) for x in range(gw) if grid[y, x] == 0]
    if not free_cells:
        return 0.0

    start = free_cells[0]
    visited = {start}
    queue = deque([start])
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < gw and 0 <= ny < gh and grid[ny,nx] == 0 and (nx,ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return len(visited) / len(free_cells)


# ── single-case evaluation ────────────────────────────────────────────────────

def evaluate_case(case_path: str, verbose: bool = False) -> Dict:
    """
    Run the full CV evaluation pipeline for one case directory.
    Returns a dict with all metrics.
    """
    case_name    = os.path.basename(case_path)
    map_path     = f"{case_path}/map.png"
    preview_path = f"{case_path}/preview.png"
    sc_path      = f"{case_path}/scenario.json"
    metrics_path = f"{case_path}/metrics.json"
    tasks_path   = f"{case_path}/tasks.json"

    result = {"case": case_name}

    # ── 1. Map parsing ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    grid, gw, gh = load_grid_auto(map_path, metrics_path)
    t_map = (time.perf_counter() - t0) * 1000  # ms

    obs_density   = float(grid.sum()) / (gw * gh)
    conn_ratio    = connectivity_ratio(grid)

    # Ground-truth density from metrics.json
    gt_density = None
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        gt_density = m.get("obstacle_density")

    density_error = abs(obs_density - gt_density) if gt_density is not None else None

    result["map"] = {
        "grid_size":       f"{gw}x{gh}",
        "obstacle_density": round(obs_density, 4),
        "gt_density":       round(gt_density, 4) if gt_density else None,
        "density_error":    round(density_error, 4) if density_error is not None else None,
        "connectivity":     round(conn_ratio, 4),
        "parse_time_ms":    round(t_map, 2),
    }

    if verbose:
        print(f"\n  [map]  {gw}x{gh}  density={obs_density:.4f}"
              f"  gt={gt_density}  err={density_error}  conn={conn_ratio:.4f}"
              f"  time={t_map:.1f}ms")

    # ── 2. Agent detection ────────────────────────────────────────────────────
    scenario   = load_scenario(sc_path)
    agents_gt  = {a["id"]: tuple(a["start"]) for a in scenario["agents"]}

    t1 = time.perf_counter()
    detected = detect_agents(preview_path, sc_path, gw, gh)
    t_agents = (time.perf_counter() - t1) * 1000

    errors      = []
    found_count = 0
    agent_rows  = []

    for d in detected:
        aid  = d["id"]
        gt   = agents_gt.get(aid)
        cv_g = d["grid"]
        err  = abs(cv_g[0] - gt[0]) + abs(cv_g[1] - gt[1]) if gt else None

        if d["found"]:
            found_count += 1
            if err is not None:
                errors.append(err)

        agent_rows.append({
            "id":       aid,
            "cv_grid":  cv_g,
            "gt_grid":  gt,
            "error":    err,
            "found_cv": d["found"],
        })

        if verbose:
            status = "✓" if d["found"] else "✗"
            print(f"  [{status}] {aid:12s}  cv={cv_g}  gt={gt}  err={err}")

    result["agents"] = {
        "total":          len(detected),
        "detected_cv":    found_count,
        "detection_rate": round(found_count / max(1, len(detected)), 3),
        "mean_error":     round(float(np.mean(errors)), 3) if errors else None,
        "max_error":      round(float(np.max(errors)),  3) if errors else None,
        "detect_time_ms": round(t_agents, 2),
        "per_agent":      agent_rows,
    }

    # ── 3. P/D marker detection (MAPD only) ───────────────────────────────────
    marker_result = None
    if scenario.get("type") == "mapd" and os.path.exists(preview_path):
        t2 = time.perf_counter()
        markers = detect_markers(preview_path, sc_path, gw, gh)
        t_markers = (time.perf_counter() - t2) * 1000

        tasks      = load_tasks(tasks_path)
        gt_markers = {}
        for task in tasks:
            gt_markers[f"{task['id']}_pickup"]  = tuple(task["pickup"])
            gt_markers[f"{task['id']}_dropoff"] = tuple(task["dropoff"])

        m_errors      = []
        m_found_count = 0
        marker_rows   = []

        for mk in markers:
            key  = f"{mk['id']}_{mk['type']}"
            gt   = gt_markers.get(key)
            cv_g = mk["grid"]
            err  = abs(cv_g[0] - gt[0]) + abs(cv_g[1] - gt[1]) if gt else None

            if mk["found"]:
                m_found_count += 1
                if err is not None:
                    m_errors.append(err)

            marker_rows.append({
                "id":      mk["id"],
                "type":    mk["type"],
                "cv_grid": cv_g,
                "gt_grid": gt,
                "error":   err,
                "found_cv": mk["found"],
            })

            if verbose:
                status = "✓" if mk["found"] else "✗"
                print(f"  [{status}] {mk['id']:10s}/{mk['type']:7s}"
                      f"  cv={cv_g}  gt={gt}  err={err}")

        marker_result = {
            "total":          len(markers),
            "detected_cv":    m_found_count,
            "detection_rate": round(m_found_count / max(1, len(markers)), 3),
            "mean_error":     round(float(np.mean(m_errors)), 3) if m_errors else None,
            "max_error":      round(float(np.max(m_errors)),  3) if m_errors else None,
            "detect_time_ms": round(t_markers, 2),
            "per_marker":     marker_rows,
        }

    result["markers"]  = marker_result
    result["timing_ms"] = {
        "map_parse":      round(t_map, 2),
        "agent_detect":   round(t_agents, 2),
        "marker_detect":  round(marker_result["detect_time_ms"], 2) if marker_result else 0.0,
        "total":          round(t_map + t_agents +
                                (marker_result["detect_time_ms"] if marker_result else 0.0), 2),
    }

    return result


# ── multi-case summary ────────────────────────────────────────────────────────

def discover_cases(data_dir: str = "data") -> List[str]:
    """Return sorted list of case directories found in data_dir."""
    if not os.path.exists(data_dir):
        return []
    return sorted([
        os.path.join(data_dir, d)
        for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and os.path.exists(os.path.join(data_dir, d, "map.png"))
    ])


def print_summary(results: List[Dict]) -> None:
    """Print a multi-section summary table to stdout."""

    # ── Map parsing table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  CV PIPELINE EVALUATION SUMMARY")
    print("=" * 72)

    print("\n┌─ 1. MAP PARSING QUALITY " + "─" * 46 + "┐")
    hdr = f"  {'Case':<30} {'Grid':>8} {'Density':>9} {'GT Dens':>9} {'DenErr':>7} {'Conn':>6} {'ms':>6}"
    print(hdr)
    print("  " + "-" * 68)
    for r in results:
        m = r["map"]
        den_err = f"{m['density_error']:.4f}" if m['density_error'] is not None else "  n/a "
        gt_den  = f"{m['gt_density']:.4f}"    if m['gt_density']    is not None else "  n/a "
        print(f"  {r['case']:<30} {m['grid_size']:>8} {m['obstacle_density']:>9.4f}"
              f" {gt_den:>9} {den_err:>7} {m['connectivity']:>6.3f} {m['parse_time_ms']:>6.1f}")

    # ── Agent detection table ──────────────────────────────────────────────
    print("\n┌─ 2. AGENT DETECTION ACCURACY " + "─" * 41 + "┐")
    hdr = f"  {'Case':<30} {'Found':>7} {'Rate':>6} {'MeanErr':>8} {'MaxErr':>7} {'ms':>6}"
    print(hdr)
    print("  " + "-" * 68)
    for r in results:
        a = r["agents"]
        me = f"{a['mean_error']:.2f}" if a['mean_error'] is not None else "  n/a"
        mx = f"{a['max_error']:.2f}"  if a['max_error']  is not None else "  n/a"
        print(f"  {r['case']:<30} {a['detected_cv']:>3}/{a['total']:<3}"
              f" {a['detection_rate']:>6.2f} {me:>8} {mx:>7} {a['detect_time_ms']:>6.1f}")

    # ── Marker detection table (MAPD cases only) ───────────────────────────
    mapd_results = [r for r in results if r["markers"] is not None]
    if mapd_results:
        print("\n┌─ 3. P/D MARKER DETECTION (MAPD) " + "─" * 37 + "┐")
        hdr = f"  {'Case':<30} {'Found':>7} {'Rate':>6} {'MeanErr':>8} {'MaxErr':>7} {'ms':>6}"
        print(hdr)
        print("  " + "-" * 68)
        for r in mapd_results:
            mk = r["markers"]
            me = f"{mk['mean_error']:.2f}" if mk['mean_error'] is not None else "  n/a"
            mx = f"{mk['max_error']:.2f}"  if mk['max_error']  is not None else "  n/a"
            print(f"  {r['case']:<30} {mk['detected_cv']:>3}/{mk['total']:<3}"
                  f" {mk['detection_rate']:>6.2f} {me:>8} {mx:>7} {mk['detect_time_ms']:>6.1f}")

    # ── Timing breakdown ───────────────────────────────────────────────────
    print("\n┌─ 4. PIPELINE TIMING BREAKDOWN (ms) " + "─" * 34 + "┐")
    hdr = f"  {'Case':<30} {'MapParse':>10} {'AgentDet':>10} {'MarkerDet':>10} {'Total':>7}"
    print(hdr)
    print("  " + "-" * 68)
    for r in results:
        t = r["timing_ms"]
        print(f"  {r['case']:<30} {t['map_parse']:>10.1f} {t['agent_detect']:>10.1f}"
              f" {t['marker_detect']:>10.1f} {t['total']:>7.1f}")

    print("\n" + "=" * 72)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CV Pipeline Evaluator")
    parser.add_argument("--case",    type=int, default=0,
                        help="0=all cases, 1-N=specific case index")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-agent and per-marker details")
    parser.add_argument("--data",    default="data",
                        help="Path to data directory")
    args = parser.parse_args()

    cases = discover_cases(args.data)
    if not cases:
        print(f"No cases found in '{args.data}/'")
        sys.exit(1)

    if args.case > 0:
        if args.case > len(cases):
            print(f"Case {args.case} not found (only {len(cases)} cases)")
            sys.exit(1)
        cases = [cases[args.case - 1]]

    print(f"Evaluating {len(cases)} case(s)...")
    results = []
    for cp in cases:
        cname = os.path.basename(cp)
        print(f"\n── {cname} ──")
        r = evaluate_case(cp, verbose=args.verbose)
        results.append(r)

    print_summary(results)

    # Save JSON report
    os.makedirs("output", exist_ok=True)
    out_path = "output/cv_evaluation.json"
    # Strip non-serialisable numpy types
    def _clean(obj):
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, dict):           return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [_clean(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_clean(results), f, indent=2)
    print(f"\nFull report saved → {out_path}")


if __name__ == "__main__":
    main()