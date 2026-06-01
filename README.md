# Platform for Multi-Agent Planning — Computer Vision Module

**HSE University · Faculty of Computer Science · 2026**  
**Author:** Anton Kushnerenko, group БПАД244  
**Supervisor:** Ivan Kopylov

---

## Overview

This project implements a computer vision module for a multi-agent planning simulator. The module reads raw images of a polygon environment and produces structured spatial data — an obstacle map and agent coordinates — that path-planning algorithms can use directly.

The full pipeline has three stages:
- **Map parsing** — converts `map.png` into a binary occupancy grid
- **Agent detection** — finds agents on `preview.png` by colour segmentation in HSV space
- **Marker detection** — locates pickup and dropoff points for MAPD delivery scenarios

Four navigation algorithms are implemented and compared: **A\***, **Bug2**, **WedgeBug**, and **FuzzyBug**.

---

## Project Structure

```
main/
├── data/
│   ├── case_001_warehouse_mapd/     # MAPD scenario, 5 agents, 10 tasks
│   ├── case_002_bottleneck_stress/  # MAPF, narrow corridor, 6 agents
│   ├── case_003_maze_mapf/          # MAPF, dense maze (51% obstacles), 6 agents
│   └── case_004_thin_walls_mapf/    # MAPF, thin walls, 8 agents
├── output/                          # JSON results from evaluation
├── map_parser.py                    # Occupancy grid parser
├── agent_detector.py                # HSV-based agent and marker detection
├── navigator.py                     # A*, Bug2, WedgeBug, FuzzyBug
├── simulator.py                     # Interactive visual simulator
├── visualizer.py                    # Static dual-window visualizer
├── cv_evaluator.py                  # Automated evaluation tool
└── main.py                          # Main entry point
```

Each `data/case_XXX/` folder contains:
| File | Description |
|------|-------------|
| `map.png` | Black-and-white obstacle map |
| `preview.png` | Rendered overhead view with agents and markers |
| `scenario.json` | Agent IDs, start positions, goals |
| `tasks.json` | Pickup/dropoff tasks for MAPD scenarios |
| `metrics.json` | Grid dimensions and obstacle statistics |

---

## Requirements

- Python 3.10+
- opencv-python
- numpy
- matplotlib

Install dependencies:
```bash
pip install opencv-python numpy matplotlib
```

---

## Usage

### Run CV pipeline on all cases
```bash
python3 main.py
```

### Run on a specific case
```bash
python3 main.py --case 1        # warehouse MAPD
python3 main.py --case 2        # bottleneck MAPF
python3 main.py --case 3        # maze MAPF
python3 main.py --case 4        # thin walls MAPF
python3 main.py --case 1 --show # with visualizer window
```

### Run the simulator
```bash
# Compare all 4 algorithms side by side (2×2 grid)
python3 simulator.py --case 1 --algo all

# Run a single algorithm
python3 simulator.py --case 1 --algo astar
python3 simulator.py --case 2 --algo bug2
python3 simulator.py --case 3 --algo wedge
python3 simulator.py --case 4 --algo fuzzy

# Adjust animation speed (ms per frame)
python3 simulator.py --case 1 --algo all --speed 40
```

**Simulator controls:** `SPACE` = pause/resume · `R` = restart · `Q` = quit

### Run evaluation
```bash
python3 cv_evaluator.py              # all cases
python3 cv_evaluator.py --case 1     # single case
python3 cv_evaluator.py --verbose    # detailed output
```

Results are saved to `output/cv_evaluation.json`.

---

## Results Summary

### Map Parsing
All four maps parsed with **zero density error** and **perfect connectivity** (1.000).

### Agent Detection
| Case | Found | Mean Error (cells) |
|------|-------|--------------------|
| case_001_warehouse_mapd | 4/5 | 22.3 |
| case_002_bottleneck_stress | 6/6 | 30.3 |
| case_003_maze_mapf | 6/6 | 55.3 |
| case_004_thin_walls_mapf | 6/8 | 61.3 |

Detection rate is 75–100%. Coordinate errors are caused by colour identity ambiguity (similar hues in the tab10 palette), not positional error.

### Navigation
| Algorithm | Case 001 | Case 002 | Case 003 | Case 004 |
|-----------|----------|----------|----------|----------|
| A* | 5/5 ✓ | 6/6 ✓ | 6/6 ✓ | 8/8 ✓ |
| Bug2 | 4/5 | 4/6 | 6/6 ✓ | 7/8 |
| WedgeBug | 4/5 | 4/6 | 6/6 ✓ | 7/8 |
| FuzzyBug | 1/5 | 1/6 | 6/6 ✓ | 7/8 |

A* always finds a path. Reactive algorithms fail on open maps due to oscillation but work well in mazes.

### Pipeline Timing
Total latency: **33–85 ms** per frame on Apple Silicon M-series.

---

## Report

The full project report is available in [`Computer_Vision_Module.pdf`](./Computer_Vision_Module.pdf).
