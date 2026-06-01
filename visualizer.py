import cv2
import numpy as np
from typing import List, Dict, Tuple

# Цвета агентов для отрисовки в BGR (OpenCV = Blue, Green, Red)
AGENT_DRAW_COLORS_BGR = [
    (219, 119,  31),   # A1  синий
    ( 14, 127, 255),   # A2  оранжевый
    ( 44, 160,  44),   # A3  зелёный
    ( 40,  39, 214),   # A4  красный
    (189, 103, 148),   # A5  фиолетовый
    ( 91,  86, 140),   # A6  коричневый
    (227, 119, 194),   # A7  розовый
    (127, 127, 127),   # A8  серый
    (188, 189,  34),   # A9  оливковый
    (188, 190,  23),   # A10 циановый
]


def build_discrete_view(grid: np.ndarray,
                        agents: List[Dict],
                        grid_w: int,
                        grid_h: int,
                        cell_size: int = 8) -> np.ndarray:
    """
    Строит дискретизированное изображение карты.

    Три «цвета»:
      Белый  (255,255,255) = свободная клетка
      Чёрный (  0,  0,  0) = стена / препятствие
      Цветной              = агент (свой цвет у каждого)
    """
    h_px = grid_h * cell_size
    w_px = grid_w * cell_size

    # Начинаем с чистой копии карты
    canvas = np.zeros((h_px, w_px, 3), dtype=np.uint8)

    for gy in range(grid_h):
        for gx in range(grid_w):
            x0, y0 = gx * cell_size, gy * cell_size
            x1, y1 = x0 + cell_size, y0 + cell_size
            if grid[gy, gx] == 1:
                color = (0, 0, 0)        # стена — чёрный
            else:
                color = (255, 255, 255)  # свободно — белый
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), color, -1)

    # Рисуем агентов поверх карты
    for i, agent in enumerate(agents):
        gx, gy = agent["grid"]

        # Защита от выхода за границы
        gx = int(np.clip(gx, 0, grid_w - 1))
        gy = int(np.clip(gy, 0, grid_h - 1))

        color = AGENT_DRAW_COLORS_BGR[i % len(AGENT_DRAW_COLORS_BGR)]

        # Центр клетки в пикселях
        cx = gx * cell_size + cell_size // 2
        cy = gy * cell_size + cell_size // 2
        radius = max(2, cell_size // 2 - 1)

        cv2.circle(canvas, (cx, cy), radius, color, -1)

        # Подпись агента (только если cell_size достаточно большой)
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
    legend = np.full((h, legend_w, 3), 40, dtype=np.uint8)  # тёмный фон

    cv2.putText(legend, "LEGEND", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    # Стена / свободно
    cv2.rectangle(legend, (10, 35), (25, 50), (0, 0, 0), -1)
    cv2.putText(legend, "Wall", (32, 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.rectangle(legend, (10, 55), (25, 70), (255, 255, 255), -1)
    cv2.putText(legend, "Free", (32, 67),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # Агенты
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
    Главная функция: показывает два окна рядом.
    Левое  — оригинальный preview (камера).
    Правое — дискретизированный вид (3 цвета).
    """
    # Оригинальное изображение
    original = cv2.imread(preview_path)

    # Дискретный вид
    discrete = build_discrete_view(grid, agents, grid_w, grid_h, cell_size)

    # Масштабируем оригинал до высоты дискретного
    target_h = discrete.shape[0]
    orig_h, orig_w = original.shape[:2]
    scale = target_h / orig_h
    orig_resized = cv2.resize(original,
                              (int(orig_w * scale), target_h))

    # Добавляем легенду к дискретному виду
    discrete_with_legend = add_legend(discrete, agents)

    # Подписи окон
    def add_title(img: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((28, img.shape[1], 3), 50, dtype=np.uint8)
        cv2.putText(bar, text, (8, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
        return np.vstack([bar, img])

    left  = add_title(orig_resized,        "Original (preview.png)")
    right = add_title(discrete_with_legend, "Discrete CV view  [W=free | B=wall | C=agent]")

    # Соединяем горизонтально
    # Если высоты разные — выравниваем
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


# ── ТЕСТ МОДУЛЯ ──────────────────────────────────────────────
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

    # Загружаем карту в КЛЕТОЧНОМ разрешении (80×60)
    grid, grid_w, grid_h = load_grid_cell_resolution(MAP_PATH, METRICS_PATH)
    print(f"Сетка: {grid_w} x {grid_h} клеток")

    # Агенты — берём координаты из scenario.json (они уже в клетках!)
    scenario = load_scenario(SCENARIO_PATH)
    agents = []
    for a in scenario["agents"]:
        agents.append({
            "id":    a["id"],
            "grid":  (a["start"][0], a["start"][1]),
            "found": True
        })

    print(f"Агентов: {len(agents)}")

    # Показываем
    show_dual_window(PREVIEW_PATH, grid, agents,
                     grid_w, grid_h, cell_size=10)