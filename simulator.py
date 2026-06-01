"""
simulator.py — Multi-Agent Path Simulator
==========================================
python3 simulator.py --case 1 --algo astar   # один алгоритм + окно восприятия
python3 simulator.py --case 1 --algo all     # сравнение 4 алгоритмов (2x2)
python3 simulator.py --list
--speed 40 (быстро) / 150 (медленно)
SPACE=start/pause  R=restart  Q=quit
"""

import cv2, numpy as np, json, os, sys, math, argparse
sys.path.insert(0, ".")
from map_parser import load_grid_auto, load_scenario
from navigator  import astar, bug2, wedgebug, fuzzybug, snap_to_free, resolve_conflicts

ALGOS      = ["astar", "bug2", "wedge", "fuzzy"]
ALGO_NAMES = {"astar": "A*", "bug2": "Bug2", "wedge": "WedgeBug", "fuzzy": "FuzzyBug"}
COLORS     = [
    (219, 119,  31), ( 14, 127, 255), ( 44, 160,  44), ( 40,  39, 214),
    (189, 103, 148), ( 91,  86, 140), (227, 119, 194), (127, 127, 127),
]
CELL = 8


# ── Загрузка ───────────────────────────────────────────────────────────────────

def discover_cases(data_dir="data"):
    if not os.path.exists(data_dir):
        return []
    return sorted(d for d in os.listdir(data_dir)
                  if os.path.isdir(f"{data_dir}/{d}")
                  and os.path.exists(f"{data_dir}/{d}/map.png"))


def load_case(case_name):
    base = f"data/{case_name}"
    metrics = f"{base}/metrics.json"
    grid, gw, gh = load_grid_auto(f"{base}/map.png",
                                   metrics if os.path.exists(metrics) else None)
    sc = load_scenario(f"{base}/scenario.json")
    agents = [{"id": a["id"],
               "pos":  snap_to_free(grid, tuple(a["start"])),
               "goal": snap_to_free(grid, tuple(a.get("goal", a["start"])))}
              for a in sc["agents"]]
    task_queue = []
    tp = f"{base}/tasks.json"
    if os.path.exists(tp):
        raw = json.load(open(tp))
        if isinstance(raw, list) and raw:
            raw.sort(key=lambda t: t.get("release_time", 0))
            task_queue = [(snap_to_free(grid, tuple(t["pickup"])),
                           snap_to_free(grid, tuple(t["dropoff"]))) for t in raw]
    return grid, gw, gh, agents, task_queue, sc.get("type", "mapf")


def plan_path(grid, start, goal, algo):
    fn = {"astar": astar, "bug2": bug2, "wedge": wedgebug, "fuzzy": fuzzybug}
    return fn[algo](grid, start, goal)


# ── Отрисовка ──────────────────────────────────────────────────────────────────

def make_base(grid):
    h, w = grid.shape
    img = np.zeros((h * CELL, w * CELL, 3), np.uint8)
    for gy in range(h):
        for gx in range(w):
            c = (30, 30, 30) if grid[gy, gx] else (230, 230, 230)
            img[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL] = c
    return img


def cc(gx, gy):
    """Центр клетки в пикселях."""
    return gx * CELL + CELL // 2, gy * CELL + CELL // 2


def rounded_rect(img, cx, cy, s, color, r=3):
    x1, y1, x2, y2 = cx-s, cy-s, cx+s, cy+s
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, -1)
    for px, py in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (px, py), r, color, -1)


def draw_robot(img, pos, color, label, goal=None, prev_pos=None):
    x, y   = pos
    cx, cy = cc(x, y)
    s = max(4, CELL - 1)
    r = max(2, s // 3)
    # Тело
    rounded_rect(img, cx, cy, s, color, r)
    # Глаз — смотрит к цели или в направлении движения
    dx, dy = 0, -1
    if goal and goal != pos:
        gx, gy = goal
        d = max(1, ((gx-x)**2 + (gy-y)**2)**0.5)
        dx, dy = (gx-x)/d, (gy-y)/d
    elif prev_pos and prev_pos != pos:
        px, py = prev_pos
        d = max(1, ((x-px)**2 + (y-py)**2)**0.5)
        dx, dy = (x-px)/d, (y-py)/d
    ex, ey = int(cx + s*0.45*dx), int(cy + s*0.45*dy)
    cv2.circle(img, (ex, ey), max(2, s//3), (255, 255, 255), -1)
    cv2.circle(img, (ex, ey), max(1, s//5), (40,  40,  40),  -1)
    # Контур
    rounded_rect(img, cx, cy, s,   (255, 255, 255), r)
    rounded_rect(img, cx, cy, s-1, color,           r)
    # Подпись
    lx, ly = cx + s + 3, cy + 5
    cv2.putText(img, label, (lx+1, ly+1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 2)
    cv2.putText(img, label, (lx,   ly),   cv2.FONT_HERSHEY_SIMPLEX, 0.35, color,   1)


def draw_goal(img, pos, color):
    cx, cy = cc(*pos)
    cv2.drawMarker(img, (cx, cy), color, cv2.MARKER_CROSS, CELL, 1)


def draw_trail(img, path, step, color):
    # Остаток — серые точки
    for t in range(min(step+1, len(path)), len(path)):
        cv2.circle(img, cc(*path[t]), 1, (180, 180, 180), -1)
    # Хвост — полупрозрачный
    tail = max(0, step - 20)
    for t in range(tail, min(step, len(path))):
        alpha = 0.1 + 0.6 * (t - tail) / max(1, step - tail)
        ov = img.copy()
        cv2.circle(ov, cc(*path[t]), max(1, CELL//2-2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def draw_perception(img_base, grid, agents, algo):
    """Правое окно: пути + алго-специфичный оверлей."""
    h, w = grid.shape
    img = np.zeros((h*CELL, w*CELL, 3), np.uint8)
    for gy in range(h):
        for gx in range(w):
            c = (20, 20, 20) if grid[gy, gx] else (245, 245, 245)
            img[gy*CELL:(gy+1)*CELL, gx*CELL:(gx+1)*CELL] = c

    for i, ag in enumerate(agents):
        color = COLORS[i % len(COLORS)]
        path  = ag.path
        step  = ag.step
        pos   = ag.current_pos()
        goal  = ag.goal
        px, py   = pos
        cx, cy   = cc(px, py)
        gcx, gcy = cc(*goal)

        # Весь путь серым
        for t in range(len(path)-1):
            cv2.line(img, cc(*path[t]), cc(*path[t+1]), (180, 180, 180), 1)
        # Пройденная часть цветом
        for t in range(min(step, len(path)-1)):
            cv2.line(img, cc(*path[t]), cc(*path[t+1]), color, 1)

        # Алго-специфичный оверлей
        if algo == "bug2":
            start_pos = ag.path[0]
            cv2.line(img, cc(*start_pos), (gcx, gcy), (200, 200, 100), 1)
            cv2.putText(img, "M", (cc(*start_pos)[0]+2, cc(*start_pos)[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 100), 1)

        elif algo == "wedge":
            angle = math.atan2(goal[1]-py, goal[0]-px)
            wlen  = 6 * CELL
            for da in [-0.7, -0.35, 0, 0.35, 0.7]:
                a  = angle + da
                ex = int(cx + wlen * math.cos(a))
                ey = int(cy + wlen * math.sin(a))
                ov = img.copy()
                cv2.line(ov, (cx, cy), (ex, ey), (100, 220, 100), 1)
                cv2.addWeighted(ov, 0.3, img, 0.7, 0, img)
            cv2.ellipse(img, (cx, cy), (wlen, wlen),
                        math.degrees(angle), -40, 40, (100, 220, 100), 1)

        elif algo == "fuzzy":
            DIRS8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
            for d8 in DIRS8:
                for t in range(1, 6):
                    nx, ny = px+d8[0]*t, py+d8[1]*t
                    if not (0 <= nx < w and 0 <= ny < h): break
                    intensity = int(255 * (1 - t/5))
                    cv2.circle(img, cc(nx, ny), 1, (intensity, intensity, 0), -1)
                    if grid[ny, nx]:
                        cv2.circle(img, cc(nx, ny), 2, (0, 0, 200), -1); break

        cv2.circle(img, (cx, cy), max(2, CELL//2), color, -1)
        cv2.circle(img, (cx, cy), max(2, CELL//2), (255, 255, 255), 1)
        cv2.drawMarker(img, (gcx, gcy), color, cv2.MARKER_CROSS, CELL, 1)

    return img


def title_bar(img, text):
    bar = np.full((26, img.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, text, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)
    return np.vstack([bar, img])


def status_bar(img, text):
    bar = np.full((26, img.shape[1], 3), 25, np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (190,190,190), 1)
    return np.vstack([img, bar])


# ── Состояние агента ───────────────────────────────────────────────────────────

class AgentState:
    def __init__(self, aid, start, goal, task_queue, is_mapd):
        self.id         = aid
        self.goal       = goal
        self.path       = [start]
        self.step       = 0
        self.phase      = None
        self.dropoff    = None
        self.task_queue = task_queue
        if is_mapd:
            self._next_task()

    def _next_task(self):
        if self.task_queue:
            pickup, dropoff = self.task_queue.pop(0)
            self.goal    = pickup
            self.dropoff = dropoff
            self.phase   = "pickup"
            return True
        self.phase = None
        return False

    def current_pos(self):
        return self.path[min(self.step, len(self.path)-1)]

    def at_goal(self):
        return self.current_pos() == self.goal and self.step >= len(self.path)-1

    def set_path(self, new_path):
        self.path = new_path if new_path else [self.current_pos()]
        self.step = 0

    def advance(self):
        if self.step < len(self.path)-1:
            self.step += 1

    def done(self):
        return self.at_goal() and self.phase is None


# ── Состояние симуляции ────────────────────────────────────────────────────────

class SimState:
    def __init__(self, grid, agents_raw, task_queue_src, sc_type, algo):
        self.grid    = grid
        self.algo    = algo
        self.is_mapd = sc_type == "mapd" and bool(task_queue_src)
        tq = list(task_queue_src)
        self.agents  = [AgentState(a["id"], a["pos"], a["goal"], tq, self.is_mapd)
                        for a in agents_raw]
        self._plan_all()

    def _plan(self, ag):
        p = plan_path(self.grid, ag.current_pos(), ag.goal, self.algo)
        ag.set_path(p)

    def _plan_all(self):
        raw = {}
        for ag in self.agents:
            p = plan_path(self.grid, ag.current_pos(), ag.goal, self.algo)
            raw[ag.id] = p if p else [ag.current_pos()]
        resolved = resolve_conflicts(raw, [ag.id for ag in self.agents])
        for ag in self.agents:
            ag.set_path(resolved[ag.id])

    def tick(self):
        # Проверяем достижение целей и переключаем фазы
        for ag in self.agents:
            if not ag.at_goal(): continue
            if ag.phase == "pickup":
                ag.goal  = ag.dropoff
                ag.phase = "dropoff"
                self._plan(ag)
            elif ag.phase == "dropoff":
                if not ag._next_task():
                    ag.phase = None
                else:
                    self._plan(ag)
        # Двигаем
        for ag in self.agents:
            ag.advance()

    def draw_sim(self, base):
        img = base.copy()
        for i, ag in enumerate(self.agents):
            color = COLORS[i % len(COLORS)]
            draw_goal(img, ag.goal, color)
            draw_trail(img, ag.path, ag.step, color)
            pos  = ag.current_pos()
            prev = ag.path[max(0, ag.step-1)]
            draw_robot(img, pos, color,
                       ag.id.replace("agent_", "A"),
                       goal=ag.goal, prev_pos=prev)
        return img

    def draw_perc(self, base):
        return draw_perception(base, self.grid, self.agents, self.algo)

    def arrived(self):
        return sum(1 for ag in self.agents if ag.done())

    def all_done(self):
        return all(ag.done() for ag in self.agents)


# ── Режим: один алгоритм ───────────────────────────────────────────────────────

def run_single(case_name, algo, speed_ms=80):
    grid, gw, gh, agents_raw, task_queue, sc_type = load_case(case_name)
    base = make_base(grid)
    n    = len(agents_raw)

    def fresh(): return SimState(grid, agents_raw, task_queue, sc_type, algo)

    sim = fresh(); paused = True; gs = 0
    print(f"{case_name} | {ALGO_NAMES[algo]} | {n} agents")
    print("SPACE=start/pause  R=restart  Q=quit")

    while True:
        sim_img  = title_bar(sim.draw_sim(base),  "Simulation")
        perc_img = title_bar(sim.draw_perc(base), f"CV Perception — {ALGO_NAMES[algo]}")

        # Выравниваем высоты
        h1, h2 = sim_img.shape[0], perc_img.shape[0]
        mh = max(h1, h2)
        if h1 < mh: sim_img  = np.vstack([sim_img,  np.zeros((mh-h1, sim_img.shape[1],  3), np.uint8)])
        if h2 < mh: perc_img = np.vstack([perc_img, np.zeros((mh-h2, perc_img.shape[1], 3), np.uint8)])

        frame = np.hstack([sim_img, perc_img])
        txt   = (f"  {ALGO_NAMES[algo]}  step:{gs}"
                 f"  goals:{sim.arrived()}/{n}"
                 f"  [SPACE=start/pause  R=restart  Q=quit]")
        frame = status_bar(frame, txt)
        cv2.imshow(f"CV Vision | {ALGO_NAMES[algo]}", frame)

        key = cv2.waitKey(speed_ms if not paused else 50) & 0xFF
        if   key in (ord("q"), ord("Q")): break
        elif key == ord(" "):             paused = not paused
        elif key in (ord("r"), ord("R")): sim = fresh(); gs = 0; paused = True

        if not paused:
            sim.tick(); gs += 1
            if sim.all_done():
                paused = True
                print(f"All done in {gs} steps!")

    cv2.destroyAllWindows()


# ── Режим: сравнение 4 алгоритмов ─────────────────────────────────────────────

def run_compare(case_name, speed_ms=80):
    grid, gw, gh, agents_raw, task_queue, sc_type = load_case(case_name)
    base = make_base(grid)
    PW, PH = gw * CELL, gh * CELL
    n = len(agents_raw)

    def fresh_all():
        return {algo: SimState(grid, agents_raw, task_queue, sc_type, algo)
                for algo in ALGOS}

    sims = fresh_all(); paused = True; gs = 0
    print(f"{case_name} | All algorithms | {n} agents")
    print("SPACE=start/pause  R=restart  Q=quit")

    while True:
        panels = []
        for algo in ALGOS:
            s   = sims[algo]
            img = s.draw_sim(base)
            # Подпись алгоритма и счётчик
            cv2.rectangle(img, (0,0), (110, 18), (30,30,30), -1)
            cv2.putText(img, ALGO_NAMES[algo], (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255,220,80), 1)
            cv2.putText(img, f"{s.arrived()}/{n}", (4, img.shape[0]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,255,180), 1)
            panels.append(cv2.resize(img, (PW, PH)))

        row1 = np.hstack([panels[0], panels[1]])
        row2 = np.hstack([panels[2], panels[3]])
        cv2.line(row1, (PW,0), (PW,PH), (80,80,80), 2)
        cv2.line(row2, (PW,0), (PW,PH), (80,80,80), 2)
        frame = np.vstack([row1, np.full((2,row1.shape[1],3),60,np.uint8), row2])
        frame = status_bar(frame,
            f"  Step:{gs}  [SPACE=start/pause  R=restart  Q=quit]")
        cv2.imshow(f"Compare | {case_name}", frame)

        key = cv2.waitKey(speed_ms if not paused else 50) & 0xFF
        if   key in (ord("q"), ord("Q")): break
        elif key == ord(" "):             paused = not paused
        elif key in (ord("r"), ord("R")): sims = fresh_all(); gs = 0; paused = True

        if not paused:
            for s in sims.values(): s.tick()
            gs += 1

    cv2.destroyAllWindows()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cases = discover_cases()
    if not cases:
        print("No cases found in data/"); sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--case",  type=int,   default=1)
    p.add_argument("--algo",  default="all",
                   choices=["all","astar","bug2","wedge","fuzzy"])
    p.add_argument("--speed", type=int,   default=80)
    p.add_argument("--list",  action="store_true")
    args = p.parse_args()

    if args.list:
        [print(f"  {i+1}: {c}") for i,c in enumerate(cases)]; sys.exit(0)

    idx  = max(1, min(args.case, len(cases))) - 1
    case = cases[idx]
    print(f"Case {idx+1}/{len(cases)}: {case}")
    if args.algo == "all": run_compare(case, args.speed)
    else:                  run_single(case, args.algo, args.speed)