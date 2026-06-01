import cv2
import numpy as np
from typing import List, Dict, Tuple

# Agent colors for rendering in BGR (OpenCV = Blue, Green, Red)
AGENT_DRAW_COLORS_BGR = [
    (219, 119,  31),   # A1  blue
    ( 14, 127, 255),   # A2  orabge
    ( 44, 160,  44),   # A3  green
    ( 40,  39, 214),   # A4  red
    (189, 103, 148),   # A5  violet
    ( 91,  86, 140),   # A6  brown
    (227, 119, 194),   # A7  pink
    (127, 127, 127),   # A8  grey
    (188, 189,  34),   # A9  plive
    (188, 190,  23),   # A10 cyan
]


def build_discrete_view(grid: np.ndarray,
                        agents: List[Dict],
                        grid_w: int,
                        grid_h: int,
                        cell_size: int = 8) -> np.ndarray:
    """
    Builds a discretized map image.

    Three "colors":
      White (255,255,255) = free cell
      Black ( 0, 0, 0) = wall / obstacle
      Color = agent (each has its own color)
    """
    h_px = grid_h * cell_size
    w_px = grid_w * cell_size

    # We start with a clean copy of the map.
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)

    for gy in range(grid_h):
        for gx in range(grid_w):
            x0, y0 = gx * cell_size, gy * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if grid[gy, gx] == 1:
                color = (0, 0, 0)        # wall - black
            else:
                color = (255, 255, 255)  # free — white
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, -1)

    # Drawing agents on top of the map
    for i, agent in enumerate(agents):
        gx, gy = agent["grid"]

        # Protection from going abroad
        gx = int(np.clip(gx, 0, grid_w - 1))
        gy = int(np.clip(gy, 0, grid_h - 1))

        color = AGENT_DRAW_COLORS_BGR[i % len(AGENT_DRAW_COLORS_BGR)]

        # The center of the cell in pixels
        cx = gx * cell_size + cell_size // 2
        cy = gy * cell_size + cell_size // 2
        radius = max(2, cell_size // 2 - 1)

        cv2.circle(canvas, (cx, cy), radius, color, -1)

        # Agent's signature (only if the cell_size is large enough)
        if cell_size >= 8:
            label = agent["id"].replace("agent_", "A")
            font_scale = 0.3
            cv2.putText(canvas, label,
                        (cx + radius + 1, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, color, 1)

    return canvas


def add_legend(img: np.ndarray, agents: List[Dict]) -> np.ndarray:
    """Добавляет легенду справа от изображения."""
    legend_w = 160
    h = img.shape[0]
    legend = np.full((h, legend_w, 3), 40, dtype=np.uint8)  # dark background

    cv2.putText(legend, "LEGEND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Wall / Free
    cv2.rectangle(legend, (10, 35), (25, 50), (0, 0, 0), -1)
    cv2.putText(legend, "Wall", (32, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.rectangle(legend, (10, 55), (25, 70), (255, 255, 255), -1)
    cv2.putText(legend, "Free", (32, 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Agents
    y = 85
    for i, agent in enumerate(agents):
        color = AGENT_DRAW_COLORS_BGR[i % len(AGENT_DRAW_COLORS_BGR)]
        label = agent["id"].replace("agent_", "A")
        gx, gy = agent["grid"]
        status = "CV" if agent.get("found") else "JSON"

        cv2.circle(legend, (18, y), 7, color, -1)
        cv2.putText(legend, f"{label} [{status}]", (32, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        cv2.putText(legend, f"  ({gx},{gy})", (32, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1)
        y += 35

    return np.hstack([img, legend])


def show_dual_window(preview_path: str,
                     grid: np.ndarray,
                     agents: List[Dict],
                     grid_w: int,
                     grid_h: int,
                     cell_size: int = 8) -> None:
    """
    Main function: Shows two windows side by side.
    On the left is the original preview (camera).
    The right one is a discretized view (3 colors).
    """
    # The original image
    original = cv2.imread(preview_path)

    # Discrete view
    discrete = build_discrete_view(grid, agents, grid_w, grid_h, cell_size)

    # Scaling the original to the height of the discrete
    target_h = discrete.shape[0]
    orig_h, orig_w = original.shape[:2]
    scale = target_h / orig_h
    orig_resized = cv2.resize(original,
                              (int(orig_w * scale), target_h))

    # Adding the legend to the discrete view
    discrete_with_legend = add_legend(discrete, agents)

    # Window captions
    def add_title(img: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((28, img.shape[1], 3), 50, dtype=np.uint8)
        cv2.putText(bar, text, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        return np.vstack([bar, img])

    left  = add_title(orig_resized,        "Original (preview.png)")
    right = add_title(discrete_with_legend, "Discrete CV view  [W=free | B=wall | C=agent]")

   # Connect horizontally
   # If the heights are different, align them
    h_l, h_r = left.shape[0], right.shape[0]
    if h_l != h_r:
        max_h = max(h_l, h_r)
        pad_l = np.zeros((max_h - h_l, left.shape[1],  3), dtype=np.uint8)
        pad_r = np.zeros((max_h - h_r, right.shape[1], 3), dtype=np.uint8)
        left  = np.vstack([left,  pad_l])
        right = np.vstack([right, pad_r])

    combined = np.hstack([left, right])

    cv2.imshow("Multi-Agent CV Vision System  |  Q = quit", combined)

    print("\n[Visualizer] Оба окна открыты. Нажми Q чтобы выйти.")
    while True:
        if cv2.waitKey(100) & 0xFF in (ord('q'), ord('Q')):
            break

    cv2.destroyAllWindows()


# === MODULE TEST ==========================================================
if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, ".")
    from map_parser     import load_grid_cell_resolution, load_scenario
    from agent_detector import detect_agents

    CASE = "case_001_warehouse_mapd"

    MAP_PATH      = f"data/{CASE}/map.png"
    PREVIEW_PATH  = f"data/{CASE}/preview.png"
    SCENARIO_PATH = f"data/{CASE}/scenario.json"
    METRICS_PATH  = f"data/{CASE}/metrics.json"

    # Download the map in CELLULAR resolution (80×60)
    grid, grid_w, grid_h = load_grid_cell_resolution(MAP_PATH, METRICS_PATH)
    print(f"Сетка: {grid_w} x {grid_h} клеток")

    # Agents - take coordinates from scenario.json (they're already in cells!)
    scenario = load_scenario(SCENARIO_PATH)
    agents = []
    for a in scenario["agents"]:
        agents.append({
            "id":    a["id"],
            "grid":  (a["start"][0], a["start"][1]),
            "found": True
        })

    print(f"Агентов: {len(agents)}")

    # Show
    show_dual_window(PREVIEW_PATH, grid, agents,
                     grid_w, grid_h, cell_size=10)
