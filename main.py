"""
main.py
=======
Entry point for the CV pipeline.

Usage:
    python3 main.py                  # run all discovered cases
    python3 main.py --case 1         # specific case (1-based index)
    python3 main.py --case 1 --show  # with visualisation
    python3 main.py --eval           # run CV evaluation (accuracy + timing)
    python3 main.py --list           # list available cases
"""

import sys, json, os, time
import cv2
import numpy as np
sys.path.insert(0, ".")

from map_parser     import load_grid_auto, load_scenario, print_grid_info
from agent_detector import detect_agents, detect_markers
from visualizer     import show_dual_window


# ── case discovery ─────────────────────────────────────────────────────────────

def discover_cases(data_dir: str = "data"):
    if not os.path.exists(data_dir):
        return []
    return sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(f"{data_dir}/{d}")
        and os.path.exists(f"{data_dir}/{d}/map.png")
    ])


# ── single case ────────────────────────────────────────────────────────────────

def run_case(case_name: str, use_cv_detection: bool = True) -> dict:
    """
    Run the full CV pipeline for one case.
    Returns result dict for reporting.
    """
    base          = f"data/{case_name}"
    map_path      = f"{base}/map.png"
    preview_path  = f"{base}/preview.png"
    sc_path       = f"{base}/scenario.json"
    metrics_path  = f"{base}/metrics.json"
    tasks_path    = f"{base}/tasks.json"

    print(f"\n{'='*55}")
    print(f"  CASE: {case_name}")
    print(f"{'='*55}")

    # ── Step 1: Map ────────────────────────────────────────────
    t0 = time.perf_counter()
    grid, gw, gh = load_grid_auto(map_path, metrics_path)
    scenario     = load_scenario(sc_path)
    t_map = (time.perf_counter() - t0) * 1000

    print_grid_info(grid, scenario)

    # ── Step 2: Agent positions ────────────────────────────────
    t1 = time.perf_counter()

    if use_cv_detection and os.path.exists(preview_path):
        agents = detect_agents(preview_path, sc_path, gw, gh)
        source = "CV detection"
    else:
        agents = [{"id": a["id"], "grid": tuple(a["start"]),
                   "found": True, "hue": None}
                  for a in scenario["agents"]]
        source = "scenario.json"

    t_agents = (time.perf_counter() - t1) * 1000

    # ── Step 3: P/D markers (MAPD only) ───────────────────────
    markers    = []
    t_markers  = 0.0
    sc_type    = scenario.get("type", "")
    if sc_type == "mapd" and os.path.exists(preview_path):
        t2 = time.perf_counter()
        markers   = detect_markers(preview_path, sc_path, gw, gh)
        t_markers = (time.perf_counter() - t2) * 1000

    # ── Step 4: Print ──────────────────────────────────────────
    print(f"\nPosition source  : {source}")
    print(f"Map parse time   : {t_map:.1f} ms")
    print(f"Agent detect time: {t_agents:.1f} ms")
    if markers:
        print(f"Marker detect time: {t_markers:.1f} ms")

    print(f"\nAgent positions:")
    for a in agents:
        tag = "CV " if a.get("found") else "GT "
        print(f"  [{tag}] {a['id']:12s}  grid={a['grid']}")

    if markers:
        found_m = sum(1 for m in markers if m["found"])
        print(f"\nP/D markers ({found_m}/{len(markers)} detected by CV):")
        for m in markers:
            tag = "CV " if m["found"] else "GT "
            print(f"  [{tag}] {m['id']:10s}/{m['type']:7s}  grid={m['grid']}")

    # ── Step 5: Export ─────────────────────────────────────────
    os.makedirs("output", exist_ok=True)
    result = {
        "case":        case_name,
        "scenario":    sc_type,
        "grid_size":   {"width": gw, "height": gh},
        "num_agents":  len(agents),
        "agents":      [{"id": a["id"],
                          "grid_x": a["grid"][0],
                          "grid_y": a["grid"][1],
                          "cv_detected": a.get("found", False)}
                         for a in agents],
        "markers":     [{"id": m["id"], "type": m["type"],
                          "grid_x": m["grid"][0], "grid_y": m["grid"][1],
                          "cv_detected": m["found"]}
                         for m in markers],
        "timing_ms":   {
            "map_parse":     round(t_map, 2),
            "agent_detect":  round(t_agents, 2),
            "marker_detect": round(t_markers, 2),
            "total":         round(t_map + t_agents + t_markers, 2),
        }
    }

    out_path = f"output/{case_name}_result.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nExport saved: {out_path}")

    return result, grid, agents, gw, gh


# ── all cases ──────────────────────────────────────────────────────────────────

def run_all_cases(cases):
    summary = []
    for case in cases:
        result, grid, agents, gw, gh = run_case(case)
        summary.append(result)

    print(f"\n{'='*68}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*68}")
    print(f"  {'Case':<32} {'Type':<22} {'N':>3} {'Total ms':>9}")
    print(f"  {'-'*64}")
    for r in summary:
        print(f"  {r['case']:<32} {r['scenario']:<22} {r['num_agents']:>3}"
              f" {r['timing_ms']['total']:>9.1f}")
    print(f"{'='*68}")

    with open("output/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nFull report: output/summary.json")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    cases = discover_cases()
    if not cases:
        print("No cases found in data/")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Multi-Agent CV Vision System")
    parser.add_argument("--case",  type=int, default=0,
                        help=f"0=all cases, 1-{len(cases)}=specific case")
    parser.add_argument("--show",  action="store_true",
                        help="Show visualisation window")
    parser.add_argument("--eval",  action="store_true",
                        help="Run CV evaluation (accuracy + timing)")
    parser.add_argument("--list",  action="store_true",
                        help="List available cases and exit")
    args = parser.parse_args()

    if args.list:
        for i, c in enumerate(cases, 1):
            print(f"  {i}: {c}")
        sys.exit(0)

    if args.eval:
        from cv_evaluator import main as eval_main
        eval_main()
        sys.exit(0)

    if args.case == 0:
        run_all_cases(cases)
    else:
        idx = max(1, min(args.case, len(cases)))
        result, grid, agents, gw, gh = run_case(cases[idx - 1])
        if args.show:
            base = f"data/{cases[idx-1]}"
            show_dual_window(
                f"{base}/preview.png",
                grid, agents, gw, gh,
                cell_size=10
            )