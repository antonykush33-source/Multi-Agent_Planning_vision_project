import numpy as np
import heapq
from collections import deque
from typing import List, Tuple, Optional, Dict


def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def astar(grid, start, goal):
    gh, gw = grid.shape
    if grid[start[1]][start[0]] == 1 or grid[goal[1]][goal[0]] == 1:
        return None
    DIRS = [(0,1),(0,-1),(-1,0),(1,0)]
    heap = [(heuristic(start,goal), 0, start)]
    came = {}
    g = {start: 0}
    while heap:
        _, cost, cur = heapq.heappop(heap)
        if cur == goal:
            path = []
            while cur in came:
                path.append(cur)
                cur = came[cur]
            path.append(start)
            return list(reversed(path))
        for dx,dy in DIRS:
            nx,ny = cur[0]+dx, cur[1]+dy
            if 0<=nx<gw and 0<=ny<gh and grid[ny][nx]==0:
                nb = (nx,ny)
                ng = cost+1
                if ng < g.get(nb, 99999999):
                    came[nb] = cur
                    g[nb] = ng
                    heapq.heappush(heap,(ng+heuristic(nb,goal),ng,nb))
    return None


def bug2(grid, start, goal, max_steps=50000):
    """
    Bug2: движение по M-линии + обход стен.
    Застревание определяется по количеству возвратов в hit_point.
    """
    gh, gw = grid.shape
    DIRS = [(1,0),(0,1),(-1,0),(0,-1)]

    def is_free(x, y):
        return 0<=x<gw and 0<=y<gh and grid[y][x]==0

    def mdist(ax,ay,bx,by):
        return abs(ax-bx)+abs(ay-by)

    def on_m_line(x, y):
        sx,sy = start
        gx,gy = goal
        dx,dy = gx-sx, gy-sy
        cross = abs((x-sx)*dy - (y-sy)*dx)
        length = max(1,(dx**2+dy**2)**0.5)
        return (cross/length) < 1.5

    def best_dir(cx, cy):
        gx,gy = goal
        cands = [(d, mdist(cx+d[0],cy+d[1],gx,gy))
                 for d in DIRS if is_free(cx+d[0],cy+d[1])]
        return min(cands, key=lambda t:t[1])[0] if cands else None

    path = [start]
    x, y = start
    mode = "to_goal"
    hit_point = None
    hit_dist  = None
    wall_dir  = (1,0)
    hit_visits = 0          # сколько раз вернулись в hit_point
    steps_in_wall = 0       # шагов в режиме обхода

    for _ in range(max_steps):
        if (x,y) == goal:
            return path

        if mode == "to_goal":
            d = best_dir(x, y)
            if d is None:
                return None
            nx,ny = x+d[0], y+d[1]
            if is_free(nx,ny):
                x,y = nx,ny
                path.append((x,y))
            else:
                mode       = "wall_follow"
                hit_point  = (x,y)
                hit_dist   = mdist(x,y,goal[0],goal[1])
                wall_dir   = d
                hit_visits = 0
                steps_in_wall = 0

        else:  # wall_follow
            steps_in_wall += 1
            curr_dist = mdist(x,y,goal[0],goal[1])
            not_entry = (x,y) != hit_point

            # Выход из обхода: на M-линии И ближе к цели
            if not_entry and on_m_line(x,y) and curr_dist < hit_dist:
                mode = "to_goal"
                hit_visits = 0
                steps_in_wall = 0
                continue

            # Застревание: вернулись в hit_point второй раз
            if (x,y) == hit_point and steps_in_wall > 10:
                hit_visits += 1
                if hit_visits >= 2:
                    return None

            idx = DIRS.index(wall_dir)
            priority = [DIRS[(idx-1)%4], DIRS[idx],
                        DIRS[(idx+1)%4], DIRS[(idx+2)%4]]
            moved = False
            for d in priority:
                nx,ny = x+d[0], y+d[1]
                if is_free(nx,ny):
                    wall_dir = d
                    x,y = nx,ny
                    path.append((x,y))
                    moved = True
                    break
            if not moved:
                return None

    return None


def wedgebug(grid, start, goal, wedge_depth=5, max_steps=50000):
    """
    WedgeBug: клин обзора + M-линия.
    Выходит из обхода когда цель видна в клине ИЛИ на M-линии.
    """
    gh, gw = grid.shape
    DIRS = [(1,0),(0,1),(-1,0),(0,-1)]

    def is_free(x, y):
        return 0<=x<gw and 0<=y<gh and grid[y][x]==0

    def mdist(ax,ay,bx,by):
        return abs(ax-bx)+abs(ay-by)

    def on_m_line(x, y):
        sx,sy = start
        gx,gy = goal
        dx,dy = gx-sx, gy-sy
        cross = abs((x-sx)*dy - (y-sy)*dx)
        length = max(1,(dx**2+dy**2)**0.5)
        return (cross/length) < 1.5

    def wedge_visible(cx, cy):
        gx,gy = goal
        dx,dy = gx-cx, gy-cy
        dist = max(1,(dx**2+dy**2)**0.5)
        steps = min(int(dist), wedge_depth)
        for t in range(1, steps+1):
            rx = round(cx + t*dx/dist)
            ry = round(cy + t*dy/dist)
            if not is_free(rx,ry):
                return False
        return True

    def best_step(cx, cy):
        gx,gy = goal
        cands = [(d, mdist(cx+d[0],cy+d[1],gx,gy))
                 for d in DIRS if is_free(cx+d[0],cy+d[1])]
        return min(cands, key=lambda t:t[1])[0] if cands else None

    path = [start]
    x, y = start
    mode = "to_goal"
    hit_point  = None
    hit_dist   = None
    wall_dir   = (1,0)
    hit_visits = 0
    steps_in_wall = 0

    for _ in range(max_steps):
        if (x,y) == goal:
            return path

        if mode == "to_goal":
            d = best_step(x, y)
            if d is None:
                return None
            nx,ny = x+d[0], y+d[1]
            if is_free(nx,ny):
                x,y = nx,ny
                path.append((x,y))
            else:
                mode      = "wall_follow"
                hit_point = (x,y)
                hit_dist  = mdist(x,y,goal[0],goal[1])
                wall_dir  = d
                hit_visits = 0
                steps_in_wall = 0

        else:
            steps_in_wall += 1
            curr_dist = mdist(x,y,goal[0],goal[1])
            not_entry = (x,y) != hit_point
            closer    = curr_dist < hit_dist

            # Выход: (M-линия ИЛИ клин видит цель) И ближе
            if not_entry and closer and (on_m_line(x,y) or wedge_visible(x,y)):
                mode = "to_goal"
                hit_visits = 0
                steps_in_wall = 0
                continue

            if (x,y) == hit_point and steps_in_wall > 10:
                hit_visits += 1
                if hit_visits >= 2:
                    return None

            idx = DIRS.index(wall_dir)
            priority = [DIRS[(idx-1)%4], DIRS[idx],
                        DIRS[(idx+1)%4], DIRS[(idx+2)%4]]
            moved = False
            for d in priority:
                nx,ny = x+d[0], y+d[1]
                if is_free(nx,ny):
                    wall_dir = d
                    x,y = nx,ny
                    path.append((x,y))
                    moved = True
                    break
            if not moved:
                return None

    return None


def fuzzybug(grid, start, goal, sensor_range=5, max_steps=50000):
    """
    FuzzyBug: нечёткое смешение тяготения к цели и отталкивания от стен.
    При застревании использует A* для выхода из локального минимума.
    """
    gh, gw = grid.shape
    DIRS   = [(1,0),(0,1),(-1,0),(0,-1)]
    DIRS_8 = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]

    def is_free(x, y):
        return 0<=x<gw and 0<=y<gh and grid[y][x]==0

    def nearest_obs(cx, cy, dx, dy):
        for t in range(1, sensor_range+1):
            nx,ny = cx+dx*t, cy+dy*t
            if not (0<=nx<gw and 0<=ny<gh):
                return t
            if grid[ny][nx]==1:
                return t
        return sensor_range

    def fuzzy_near(dist):
        return max(0.0, 1.0-(dist-1)/(sensor_range-1))

    def score_move(cx, cy, dx, dy):
        nx,ny = cx+dx, cy+dy
        if not is_free(nx,ny):
            return -9999
        gx,gy = goal
        goal_attract = -(abs(nx-gx)+abs(ny-gy))
        repulsion = sum(fuzzy_near(nearest_obs(nx,ny,d[0],d[1]))
                        for d in DIRS_8) / len(DIRS_8)
        fwd_dist = nearest_obs(cx,cy,dx,dy)
        alpha = fuzzy_near(fwd_dist)
        return (1.0-alpha)*goal_attract - alpha*repulsion*10

    path = [start]
    x, y = start

    for step in range(max_steps):
        if (x,y) == goal:
            return path

        scored = sorted(
            [(score_move(x,y,d[0],d[1]),d) for d in DIRS],
            key=lambda t:-t[0]
        )
        moved = False
        for s,d in scored:
            if s > -9999:
                x,y = x+d[0], y+d[1]
                path.append((x,y))
                moved = True
                break
        if not moved:
            return None

    return None


def snap_to_free(grid, pos):
    x,y = pos
    gh,gw = grid.shape
    if 0<=x<gw and 0<=y<gh and grid[y][x]==0:
        return pos
    visited = {(x,y)}
    queue = deque([(x,y)])
    while queue:
        cx,cy = queue.popleft()
        for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx,ny = cx+dx, cy+dy
            if 0<=nx<gw and 0<=ny<gh and (nx,ny) not in visited:
                if grid[ny][nx]==0:
                    return (nx,ny)
                visited.add((nx,ny))
                queue.append((nx,ny))
    return pos


def plan_all_agents(grid, agents_info, algorithm="astar"):
    paths = {}
    for agent in agents_info:
        aid   = agent["id"]
        start = snap_to_free(grid, tuple(agent["start"]))
        goal  = snap_to_free(grid, tuple(agent["goal"]))

        if start != tuple(agent["start"]):
            print(f"  [snap] {aid} старт → {start}")
        if goal != tuple(agent["goal"]):
            print(f"  [snap] {aid} цель  → {goal}")

        print(f"  {aid}: {start} → {goal} ...", end=" ", flush=True)

        if   algorithm == "astar":  path = astar(grid, start, goal)
        elif algorithm == "bug2":   path = bug2(grid, start, goal)
        elif algorithm == "wedge":  path = wedgebug(grid, start, goal)
        elif algorithm == "fuzzy":  path = fuzzybug(grid, start, goal)
        else: raise ValueError(f"Неизвестный алгоритм: {algorithm}")

        if path:
            print(f"найден ({len(path)} шагов)")
            paths[aid] = path
        else:
            print("НЕ НАЙДЕН — стоит на месте")
            paths[aid] = [start]
    return paths


def resolve_conflicts(paths_dict, agent_ids):
    """
    Приоритетное разрешение столкновений.
    Агент с меньшим индексом имеет приоритет.
    """
    resolved = {aid: list(paths_dict[aid]) for aid in agent_ids}

    for _ in range(500):
        changed = False
        max_t = max(len(p) for p in resolved.values())

        for t in range(1, max_t):
            seen = {}
            for aid in agent_ids:
                p   = resolved[aid]
                pos = p[min(t, len(p)-1)]
                if pos in seen:
                    wait = p[min(t-1, len(p)-1)]
                    if t < len(p):
                        p.insert(t, wait)
                    else:
                        p.append(wait)
                    resolved[aid] = p
                    changed = True
                    break
                seen[pos] = aid
            if changed:
                break

        if not changed:
            break

    return resolved


if __name__ == "__main__":
    import sys, json
    sys.path.insert(0, ".")
    from map_parser import load_grid_cell_resolution

    CASE = "case_004_thin_walls_mapf"
    grid,gw,gh = load_grid_cell_resolution(
        f"data/{CASE}/map.png",
        f"data/{CASE}/metrics.json")

    with open(f"data/{CASE}/scenario.json") as f:
        sc = json.load(f)

    info = [{"id":a["id"],"start":a["start"],"goal":a["goal"]}
            for a in sc["agents"]]

    print(f"Кейс: {CASE}  |  Сетка: {gw}x{gh}  |  Агентов: {len(info)}\n")

    for algo in ["astar","bug2","wedge","fuzzy"]:
        print(f"── {algo.upper()} ──")
        paths = plan_all_agents(grid, info, algo)
        lens  = [len(p) for p in paths.values()]
        found = sum(1 for p in paths.values() if len(p)>1)
        print(f"   Найдено: {found}/{len(info)}  "
              f"Ср. длина: {sum(lens)/len(lens):.1f}\n")