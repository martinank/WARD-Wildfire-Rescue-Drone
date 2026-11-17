import argparse
import heapq
import math
import random
import time
from collections import deque

import pygame

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
GRID = 30                       # map is GRID x GRID cells
TW, TH = 32, 16                 # isometric tile width / height (pixels)
HP = 6                          # pixels per terrain height level
W, H = 1300, 760                # window size
PANEL = 310                     # HUD panel width
MAPW = W - PANEL
OX, OY = MAPW // 2, 150         # screen origin of the isometric map
FIRE_TICK = 0.5                 # seconds between fire-spread steps (at 1x speed)
DRONE_SPEED = 4.0               # cells per second
NUM_VICTIMS = 3

# terrain types
GRASS, TREE, ROCK, WATER, BRUSH, PAD = range(6)
# fire states
UNBURNT, BURNING, BURNT = range(3)

INF = float("inf")
SQ2 = math.sqrt(2)
DIRS = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]

FLAMMABLE = (GRASS, TREE, BRUSH)
IGNITE_P = {GRASS: 0.12, TREE: 0.11, BRUSH: 0.70}        # BRUSH = "fire fuse"
BURN_TICKS = {GRASS: (2, 3), TREE: (6, 10), BRUSH: (1, 2)}
MOVE_COST = {GRASS: 1.0, TREE: 1.6, ROCK: INF, WATER: 1.0, BRUSH: 1.0, PAD: 1.0}
SMOKE_COST = 3.0                                          # reduced visibility
DANGER = {1: 14.0, 2: 7.0, 3: 3.0, 4: 1.0}                # penalty by distance to fire
HEURISTICS = ["octile", "risk-aware", "manhattan", "dijkstra"]

BASE_COL = {GRASS: (88, 132, 60), TREE: (60, 100, 48), ROCK: (104, 104, 108),
            WATER: (38, 94, 150), BRUSH: (158, 138, 66), PAD: (118, 124, 132)}


# ----------------------------------------------------------------------------
# Grid: terrain, fire spread, hazard maps
# ----------------------------------------------------------------------------
class World:
    def __init__(self, seed):
        self.seed = seed
        r = self.rng = random.Random(seed)
        n = GRID

        def grid(v):
            return [[v] * n for _ in range(n)]

        self.terrain, self.height, self.state = grid(GRASS), grid(0), grid(UNBURNT)
        self.burn_left, self.fire_dist = grid(0), grid(99)
        self.risk, self.blocked, self.smoke = grid(0.0), grid(False), grid(False)
        self.tint = [[r.randint(-9, 9) for _ in range(n)] for _ in range(n)]
        self.scale = [[0.8 + 0.5 * r.random() for _ in range(n)] for _ in range(n)]
        self.burning = set()
        self.wind = r.choice(DIRS)
        self.base, self.victim_cells = (3, n - 4), []
        self.ok = self._generate()
        self.recompute_hazards()

    # -- generation ---------------------------------------------------------
    def _generate(self):
        r, n, T = self.rng, GRID, self.terrain
        p = [r.uniform(0, 6.28) for _ in range(4)]
        for y in range(n):
            for x in range(n):
                hv = (2.2 + 1.4 * math.sin(x * .21 + p[0]) + 1.3 * math.cos(y * .18 + p[1])
                      + .9 * math.sin((x + y) * .12 + p[2]))
                self.height[y][x] = max(0, min(5, int(hv)))
                dens = .5 + .5 * math.sin(x * .33 + p[2]) * math.cos(y * .29 + p[3])
                T[y][x] = TREE \
                    if r.random() < .15 + .55 * dens \
                    else GRASS
                if r.random() < .03:
                    T[y][x] = ROCK                      # boulders: impassable
        for _ in range(2):                              # ponds (safe zones)
            cx, cy = r.randint(6, n - 7), r.randint(6, n - 7)
            rx, ry = r.uniform(1.5, 3), r.uniform(1.5, 3)
            for y in range(n):
                for x in range(n):
                    if ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1:
                        T[y][x] = WATER
                        self.height[y][x] = 0
        bx, by = self.base                              # helipad
        hb = self.height[by][bx]
        for y in range(by - 2, by + 3):
            for x in range(bx - 2, bx + 3):
                if 0 <= x < n and 0 <= y < n:
                    T[y][x] = PAD if max(abs(x - bx), abs(y - by)) <= 1 else GRASS
                    self.height[y][x] = hb
        cells = []                                      # victims
        for _ in range(600):
            if len(cells) == NUM_VICTIMS:
                break
            x, y = r.randint(3, n - 4), r.randint(3, n - 4)
            if T[y][x] in (WATER, ROCK, PAD) or math.hypot(x - bx, y - by) < 14:
                continue
            if any(math.hypot(x - cx, y - cy) < 8 for cx, cy in cells):
                continue
            cells.append((x, y))
        if len(cells) < NUM_VICTIMS:
            return False
        for x, y in cells:                              # small clearing around each
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if T[y + dy][x + dx] not in (WATER, PAD):
                        T[y + dy][x + dx] = GRASS
        self.victim_cells = cells
        starts = []                                     # ignition points
        for _ in range(600):
            if len(starts) == 2:
                break
            x, y = r.randint(2, n - 3), r.randint(2, n - 3)
            if T[y][x] not in FLAMMABLE or math.hypot(x - bx, y - by) < 15:
                continue
            if any(math.hypot(x - vx, y - vy) < 10 for vx, vy in cells):
                continue
            if any(math.hypot(x - sx, y - sy) < 10 for sx, sy in starts):
                continue
            starts.append((x, y))
        if len(starts) < 2:
            return False
        for i, (sx, sy) in enumerate(starts):
            vx, vy = cells[i]
            self._fuse(sx, sy, vx, vy)
            self.ignite(sx, sy)
        seen, dq = {self.base}, deque([self.base])      # all victims reachable?
        while dq:
            x, y = dq.popleft()
            for dx, dy in DIRS[1::2]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and (nx, ny) not in seen and T[ny][nx] != ROCK:
                    seen.add((nx, ny))
                    dq.append((nx, ny))
        return all(c in seen for c in cells)

    def _fuse(self, x0, y0, x1, y1):
        r, x, y = self.rng, float(x0), float(y0)
        for _ in range(max(0, min(int(math.hypot(x1 - x0, y1 - y0)) - 4, 10))):
            dx, dy = x1 - x, y1 - y
            d = math.hypot(dx, dy) or 1
            x += dx / d + r.uniform(-.5, .5)
            y += dy / d + r.uniform(-.5, .5)
            cx, cy = int(round(x)), int(round(y))
            if 0 <= cx < GRID and 0 <= cy < GRID and self.terrain[cy][cx] in (GRASS, TREE):
                self.terrain[cy][cx] = BRUSH

    # -- fire ---------------------------------------------------------------
    def ignite(self, x, y):
        if self.state[y][x] != UNBURNT or self.terrain[y][x] not in FLAMMABLE:
            return False
        self.state[y][x] = BURNING
        self.burn_left[y][x] = self.rng.randint(*BURN_TICKS[self.terrain[y][x]])
        self.burning.add((x, y))
        return True

    def step_fire(self):
        r, wx, wy = self.rng, *self.wind
        wn = math.hypot(wx, wy)
        new, done = [], []
        for x, y in list(self.burning):
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < GRID and 0 <= ny < GRID):
                    continue
                if self.state[ny][nx] != UNBURNT or self.terrain[ny][nx] not in FLAMMABLE:
                    continue
                p = IGNITE_P[self.terrain[ny][nx]]
                if dx and dy:
                    p *= 0.7
                p *= max(0.25, 1 + 0.9 * (dx * wx + dy * wy) / (math.hypot(dx, dy) * wn))  # wind
                p *= max(0.5, 1 + 0.12 * (self.height[ny][nx] - self.height[y][x]))        # uphill
                if r.random() < p:
                    new.append((nx, ny))
            if self.terrain[y][x] == TREE and r.random() < 0.015:                          # spotting
                d = r.randint(3, 6)
                sx, sy = int(x + wx / wn * d + r.randint(-1, 1)), int(y + wy / wn * d + r.randint(-1, 1))
                if 0 <= sx < GRID and 0 <= sy < GRID:
                    new.append((sx, sy))
            self.burn_left[y][x] -= 1
            if self.burn_left[y][x] <= 0:
                done.append((x, y))
        for c in new:
            self.ignite(*c)
        for x, y in done:
            self.state[y][x] = BURNT
            self.burning.discard((x, y))

    def recompute_hazards(self):
        """Distance-to-fire map, smoke plume, risk penalty and blocked cells."""
        n = GRID
        fd = [[99] * n for _ in range(n)]
        dq = deque()
        for x, y in self.burning:
            fd[y][x] = 0
            dq.append((x, y))
        while dq:
            x, y = dq.popleft()
            d = fd[y][x]
            if d >= 6:
                continue
            for dx, dy in DIRS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n and fd[ny][nx] > d + 1:
                    fd[ny][nx] = d + 1
                    dq.append((nx, ny))
        self.fire_dist = fd
        sm = [[False] * n for _ in range(n)]
        wx, wy = self.wind
        for x, y in self.burning:                       # smoke plume drifts downwind
            for k in range(1, 5):
                for side in (-1, 0, 1):
                    if side and k < 2:
                        continue
                    px, py = x + wx * k - wy * side, y + wy * k + wx * side
                    if 0 <= px < n and 0 <= py < n:
                        sm[py][px] = True
        self.smoke = sm
        for y in range(n):
            for x in range(n):
                d, t, st = fd[y][x], self.terrain[y][x], self.state[y][x]
                rk = DANGER.get(d, 0.0) + (SMOKE_COST if sm[y][x] else 0.0)
                fuse = t == BRUSH and st == UNBURNT
                if fuse and d <= 6:
                    rk += 6.0                           # a live fuse is about to flash over
                self.risk[y][x] = rk
                self.blocked[y][x] = t == ROCK or st == BURNING or (fuse and d <= 2)

    def cost_of(self, x, y):
        """Cost of entering a cell (None = blocked). Always >= 1."""
        if self.blocked[y][x]:
            return None
        base = 1.0 if self.state[y][x] == BURNT else MOVE_COST[self.terrain[y][x]]
        return base + self.risk[y][x]


def make_world(seed):
    while True:
        w = World(seed)
        if w.ok:
            return w
        seed += 1


# ----------------------------------------------------------------------------
# A* search
# ----------------------------------------------------------------------------
def make_heuristic(name, world):
    def octile(x, y, gx, gy):
        dx, dy = abs(x - gx), abs(y - gy)
        return dx + dy + (SQ2 - 2) * min(dx, dy)

    if name == "dijkstra":                  # h = 0  -> uniform-cost search
        return lambda x, y, gx, gy: 0.0
    if name == "manhattan":                 # over-estimates on diagonals (not admissible)
        return lambda x, y, gx, gy: abs(x - gx) + abs(y - gy)
    if name == "risk-aware":
        risk = world.risk

        def risk_aware(x, y, gx, gy):
            base = octile(x, y, gx, gy)
            k = min(6, max(abs(x - gx), abs(y - gy)))
            if k <= 1:
                return base
            tot = 0.0
            for i in range(1, k):                    # sample hazard along the straight line
                tot += risk[y + round((gy - y) * i / k)][x + round((gx - x) * i / k)]
            return base * (1.0 + 0.1 * tot / (k - 1))
        return risk_aware
    return octile                                    # admissible & consistent (default)


def astar(world, start, goal, hname="octile"):
    """A* on an 8-connected grid. f(n) = g(n) + h(n).
    g includes terrain cost + fire danger + smoke, so the cheapest path is also the safest."""
    t0 = time.perf_counter()
    h = make_heuristic(hname, world)
    gx, gy = goal
    counter = 0
    heap = [(h(*start, gx, gy), counter, start)]
    g, came, closed, expanded = {start: 0.0}, {}, set(), []
    while heap:
        _, _, cur = heapq.heappop(heap)
        if cur in closed:
            continue
        closed.add(cur)
        expanded.append(cur)
        if cur == goal:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            path.reverse()
            return dict(found=True, path=path, cost=g[goal], expanded=len(expanded),
                        explored=expanded, ms=(time.perf_counter() - t0) * 1000)
        cx, cy = cur
        for dx, dy in DIRS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID and 0 <= ny < GRID):
                continue
            c = world.cost_of(nx, ny)
            if c is None:
                continue
            if dx and dy:
                if world.blocked[cy][nx] or world.blocked[ny][cx]:   # no corner cutting
                    continue
                c *= SQ2
            ng = g[cur] + c
            if ng < g.get((nx, ny), INF):
                g[(nx, ny)] = ng
                came[(nx, ny)] = cur
                counter += 1
                heapq.heappush(heap, (ng + h(nx, ny, gx, gy), counter, (nx, ny)))
    return dict(found=False, path=[], cost=INF, expanded=len(expanded), explored=expanded,
                ms=(time.perf_counter() - t0) * 1000)


# ----------------------------------------------------------------------------
# Rescue mission (drone agent / state machine)
# ----------------------------------------------------------------------------
class Victim:
    def __init__(self, idx, cell):
        self.idx, self.cell, self.status = idx, cell, "hidden"   # hidden/located/carried/rescued/lost


STATE_TEXT = {"SCAN": "Thermal scan: locating victims", "PLAN": "Planning route (A*)",
              "TO_VICTIM": "Flying to victim", "RESCUE": "Winching victim aboard",
              "TO_BASE": "Extracting - returning to base", "DROP": "Landing at helipad",
              "HOLD": "HOLD: no safe path, waiting", "DONE": "MISSION COMPLETE",
              "CRASHED": "DRONE LOST"}


class Mission:
    def __init__(self, world, heuristic="octile"):
        self.w, self.heuristic, self.base = world, heuristic, world.base
        self.victims = [Victim(i, c) for i, c in enumerate(world.victim_cells)]
        self.cell, self.nxt, self.prog, self.seg = self.base, None, 0.0, 1.0
        self.path, self.pos = [], (float(self.base[0]), float(self.base[1]))
        self.health, self.state, self.hold_state = 100.0, "SCAN", None
        self.timer, self.scan_r, self.clock, self.rotor = 0.0, 0.0, 0.0, 0.0
        self.carrying = self.target = None
        self.explored, self.log = [], []
        self.stats = dict(searches=0, total_nodes=0, last_nodes=0, last_cost=0.0,
                          last_len=0, last_ms=0.0, reroutes=0)
        self.say("Drone launched - starting thermal scan")

    def say(self, msg):
        self.log.append(f"[{self.clock:5.1f}s] {msg}")
        del self.log[:-6]

    # -- planning -------------------------------------------------------------
    def plan_start(self):
        return self.nxt if self.nxt else self.cell

    def search(self, start, goal):
        res = astar(self.w, start, goal, self.heuristic)
        st = self.stats
        st["searches"] += 1
        st["total_nodes"] += res["expanded"]
        return res

    def record(self, res):
        st = self.stats
        st["last_nodes"], st["last_ms"] = res["expanded"], res["ms"]
        st["last_cost"], st["last_len"] = res["cost"], len(res["path"]) - 1
        self.explored = res["explored"]

    def plan_next(self):
        cands = [v for v in self.victims if v.status == "located"]
        if not cands:
            return self.finish()
        best = None
        for v in cands:
            res = self.search(self.plan_start(), v.cell)
            if res["found"] and (best is None or res["cost"] < best[1]["cost"]):
                best = (v, res)
        if best is None:
            self.state, self.hold_state = "HOLD", "PLAN"
            self.say("No safe path to any victim - holding")
            return
        self.target, res = best[0], best[1]
        self.record(res)
        self.path = res["path"][1:]
        self.state = "TO_VICTIM"
        self.say(f"A* target: victim #{self.target.idx + 1} (cost {res['cost']:.1f}, "
                 f"{res['expanded']} nodes)")

    def finish(self):
        self.state = "DONE"
        n = sum(v.status == "rescued" for v in self.victims)
        self.say(f"Mission complete: {n}/{len(self.victims)} rescued")
        print(f"\n=== MISSION RESULT ({self.heuristic}) ===")
        for v in self.victims:
            print(f"  victim #{v.idx + 1} at {v.cell}: {v.status}")
        s = self.stats
        print(f"  time {self.clock:.1f}s | searches {s['searches']} | nodes explored "
              f"{s['total_nodes']} | reroutes {s['reroutes']} | drone health {self.health:.0f}%")

    def replan(self):
        """Called after every fire step: re-run A* so the route adapts to the spreading fire."""
        st = self.state
        if st == "HOLD" and self.hold_state == "PLAN":
            self.state = "PLAN"
            return
        cur = self.hold_state if st == "HOLD" else st
        if cur not in ("TO_VICTIM", "TO_BASE"):
            return
        if cur == "TO_VICTIM":
            if self.target.status != "located":
                self.state = "PLAN"
                return
            goal = self.target.cell
        else:
            goal = self.base
        res = self.search(self.plan_start(), goal)
        if res["found"]:
            self.record(res)
            if res["path"][1:] != self.path:
                self.stats["reroutes"] += 1
            self.path = res["path"][1:]
            if st == "HOLD":
                self.say("Safe path found - resuming")
            self.state, self.hold_state = cur, None
        elif st != "HOLD":
            self.state, self.hold_state = "HOLD", cur
            self.say("Path blocked by fire - holding position")

    def on_fire_tick(self):
        for v in self.victims:
            if v.status in ("hidden", "located") and self.w.state[v.cell[1]][v.cell[0]] == BURNING:
                v.status = "lost"
                self.say(f"Victim #{v.idx + 1} lost to fire")
        if self.state not in ("DONE", "CRASHED"):
            self.replan()

    # -- per-frame update -------------------------------------------------------
    def update(self, dt):
        s = self.state
        if s in ("DONE", "CRASHED"):
            return
        self.clock += dt
        self.rotor += dt * 40
        px, py = (min(GRID - 1, max(0, int(round(v)))) for v in self.pos)
        d = self.w.fire_dist[py][px]
        self.health -= (30 if d == 0 else 4 if d == 1 else 0) * dt     # heat damage
        if self.health <= 0:
            self.state = "CRASHED"
            if self.carrying:
                self.carrying.status = "lost"
            self.say("Drone destroyed by heat!")
            return
        if s == "SCAN":
            self.scan_r += 20 * dt
            for v in self.victims:
                if v.status == "hidden" and math.hypot(v.cell[0] - self.base[0],
                                                       v.cell[1] - self.base[1]) <= self.scan_r:
                    v.status = "located"
                    self.say(f"Victim #{v.idx + 1} located at {v.cell}")
            if self.scan_r > 44:
                self.state = "PLAN"
        elif s == "PLAN":
            self.plan_next()
        elif s in ("TO_VICTIM", "TO_BASE"):
            self.fly(dt)
        elif s == "RESCUE":
            self.timer -= dt
            if self.timer <= 0:
                self.pickup()
        elif s == "DROP":
            self.timer -= dt
            if self.timer <= 0:
                self.carrying.status = "rescued"
                self.say(f"Victim #{self.carrying.idx + 1} delivered safely")
                self.carrying, self.state = None, "PLAN"

    def fly(self, dt):
        if self.nxt is None:
            if not self.path:
                return self.arrive()
            self.nxt = self.path.pop(0)
            self.prog = 0.0
            self.seg = math.hypot(self.nxt[0] - self.cell[0], self.nxt[1] - self.cell[1])
        self.prog += DRONE_SPEED * dt / self.seg
        if self.prog >= 1:
            self.cell, self.nxt = self.nxt, None
            self.pos = (float(self.cell[0]), float(self.cell[1]))
        else:
            a, b, t = self.cell, self.nxt, self.prog
            self.pos = (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)

    def arrive(self):
        if self.state == "TO_VICTIM":
            if self.target.status != "located":
                self.state = "PLAN"
                return
            self.state, self.timer = "RESCUE", 2.0
            self.say(f"Hovering over victim #{self.target.idx + 1} - winching")
        else:
            self.state, self.timer = "DROP", 1.2

    def pickup(self):
        v = self.target
        if v.status != "located":
            self.state = "PLAN"
            return
        v.status, self.carrying = "carried", v
        res = self.search(self.cell, self.base)
        if res["found"]:
            self.record(res)
            self.path, self.state = res["path"][1:], "TO_BASE"
            self.say(f"Victim #{v.idx + 1} aboard - extracting (cost {res['cost']:.1f})")
        else:
            self.state, self.hold_state = "HOLD", "TO_BASE"
            self.say("Victim aboard but no safe path home - holding")


# ----------------------------------------------------------------------------
# Projection helpers (isometric)
# ----------------------------------------------------------------------------
def top(w, x, y):
    """Screen position of the centre of a cell's top face."""
    return OX + (x - y) * TW / 2, OY + (x + y) * TH / 2 - w.height[y][x] * HP


def frac_top(w, fx, fy):
    x0, y0 = min(GRID - 2, max(0, int(fx))), min(GRID - 2, max(0, int(fy)))
    tx, ty = fx - x0, fy - y0
    hh = (w.height[y0][x0] * (1 - tx) * (1 - ty) + w.height[y0][x0 + 1] * tx * (1 - ty)
          + w.height[y0 + 1][x0] * (1 - tx) * ty + w.height[y0 + 1][x0 + 1] * tx * ty)
    return OX + (fx - fy) * TW / 2, OY + (fx + fy) * TH / 2 - hh * HP


def screen_to_cell(w, mx, my):
    a, b = (mx - OX) / (TW / 2), (my - OY) / (TH / 2)
    gx, gy = int(round((a + b) / 2)), int(round((b - a) / 2))
    best, bd = None, 1e9
    for y in range(gy - 3, gy + 4):
        for x in range(gx - 3, gx + 4):
            if 0 <= x < GRID and 0 <= y < GRID:
                cx, cy = top(w, x, y)
                d = abs(cx - mx) / (TW / 2) + abs(cy - my) / (TH / 2)
                if d < bd:
                    best, bd = (x, y), d
    return best if bd <= 1.0 else None


def lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c)


# ----------------------------------------------------------------------------
# Simulation = world + mission + particles
# ----------------------------------------------------------------------------
class Simulation:
    def __init__(self, seed, heuristic):
        self.seed = seed
        self.world = make_world(seed)
        self.mission = Mission(self.world, heuristic)
        self.fire_timer, self.time = 0.0, 0.0
        self.smoke, self.embers = [], []
        self.fx = random.Random()

    def update(self, dt):
        self.time += dt
        self.mission.update(dt)
        self.fire_timer += dt
        while self.fire_timer >= FIRE_TICK:
            self.fire_timer -= FIRE_TICK
            self.world.step_fire()
            self.world.recompute_hazards()
            self.mission.on_fire_tick()

    def update_particles(self, dt):
        w, fx = self.world, self.fx
        wx, wy = w.wind
        dxs, dys = (wx - wy) * TW / 2, (wx + wy) * TH / 2
        nrm = math.hypot(dxs, dys) or 1
        vx, vy = dxs / nrm * 14, dys / nrm * 14
        for x, y in w.burning:
            cx, cy = top(w, x, y)
            tree = w.terrain[y][x] == TREE
            if len(self.smoke) < 300 and fx.random() < dt * (1.4 if tree else .6):
                self.smoke.append([cx + fx.uniform(-4, 4), cy - 10, vx + fx.uniform(-4, 4),
                                   vy - fx.uniform(18, 30), 0.0, fx.uniform(2.5, 4.5),
                                   fx.uniform(6, 10)])
            if len(self.embers) < 160 and fx.random() < dt * (2.2 if tree else 1.0):
                self.embers.append([cx + fx.uniform(-4, 4), cy - 10, vx * 1.4 + fx.uniform(-12, 12),
                                    vy - fx.uniform(25, 55), 0.0, fx.uniform(1, 2.2)])
        for lst, life in ((self.smoke, 5), (self.embers, 5)):
            for p in lst:
                p[0] += p[2] * dt
                p[1] += p[3] * dt
                p[4] += dt
            lst[:] = [p for p in lst if p[4] < p[5]]


# ----------------------------------------------------------------------------
# Renderer
# ----------------------------------------------------------------------------
class Renderer:
    def __init__(self, screen):
        self.s = screen
        names = "consolas,dejavusansmono,menlo,couriernew"
        self.f = pygame.font.SysFont(names, 14)
        self.fb = pygame.font.SysFont(names, 18, bold=True)
        self.fs = pygame.font.SysFont(names, 12)
        self.ov = pygame.Surface((MAPW, H), pygame.SRCALPHA)
        self.dark = pygame.Surface((MAPW, H), pygame.SRCALPHA)
        self.bg = pygame.Surface((MAPW, H))
        for y in range(H):
            pygame.draw.line(self.bg, lerp((10, 12, 26), (34, 22, 24), y / H), (0, y), (MAPW, y))
        self.cut, self.add, self.puffs = {}, {}, {}
        self.show_path, self.show_explored, self.t, self.w = True, False, 0.0, None

    # -- cached light sprites -----------------------------------------------
    def glow_cut(self, r):
        if r not in self.cut:
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            for i in range(r, 0, -3):
                pygame.draw.circle(s, (0, 0, 0, int(200 * (1 - i / r) ** 1.4)), (r, r), i)
            self.cut[r] = s
        return self.cut[r]

    def glow_add(self, r, col):
        key = (r, col)
        if key not in self.add:
            s = pygame.Surface((r * 2, r * 2))
            for i in range(r, 0, -3):
                f = (1 - i / r) ** 1.7
                pygame.draw.circle(s, tuple(int(c * f) for c in col), (r, r), i)
            self.add[key] = s
        return self.add[key]

    def puff(self, r):
        if r not in self.puffs:
            s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            for i in range(r, 0, -2):
                pygame.draw.circle(s, (62, 58, 58, int(150 * (1 - i / r) ** 1.1)), (r, r), i)
            self.puffs[r] = s
        return self.puffs[r]

    def text(self, msg, x, y, col=(220, 224, 232), font=None):
        self.s.blit((font or self.f).render(msg, True, col), (x, y))

    @staticmethod
    def diamond(cx, cy, k=1.0):
        return [(cx, cy - TH / 2 * k), (cx + TW / 2 * k, cy), (cx, cy + TH / 2 * k), (cx - TW / 2 * k, cy)]

    # -- terrain -------------------------------------------------------------
    def tile_color(self, x, y):
        w = self.w
        t, st, h = w.terrain[y][x], w.state[y][x], w.height[y][x]
        if st == BURNT:
            c = (40, 36, 34)
        elif st == BURNING:
            c = lerp((170, 48, 14), (255, 150, 32), .5 + .5 * math.sin(self.t * 9 + x * 1.7 + y * 2.3))
        else:
            c = BASE_COL[t]
            if t == WATER:
                c = lerp(c, (90, 160, 215), .2 + .2 * math.sin(self.t * 2 + x + y * .7))
            d = w.fire_dist[y][x]
            if d <= 3:
                c = lerp(c, (200, 90, 30), (4 - d) / 4 * .35)         # heat glow on the ground
        k = 1 + .045 * h
        return tuple(max(0, min(255, int(v * k) + w.tint[y][x])) for v in c)

    def flames(self, cx, cy, size, seed):
        s = self.s
        for k in range(3):
            ph = self.t * 8 + seed + k * 2.1
            hg = size * (.8 + .4 * math.sin(ph)) * (1 - .2 * k)
            wd = size * .45 * (1 - .15 * k)
            ox = cx + (k - 1) * size * .35 + math.sin(ph * 1.3) * 2
            tipx = ox + math.sin(ph) * 3
            for sc, col in ((1.0, (225, 60, 18)), (.7, (255, 150, 30)), (.4, (255, 230, 120))):
                pygame.draw.polygon(s, col, [(ox - wd * sc, cy), (ox - wd * sc * .4, cy - hg * .55 * sc),
                                             (tipx, cy - hg * sc), (ox + wd * sc * .4, cy - hg * .55 * sc),
                                             (ox + wd * sc, cy)])

    def draw_tree(self, x, y, cx, cy, st):
        w, s = self.w, self.s
        sc = w.scale[y][x]
        if st == BURNT:                                               # charred skeleton
            pygame.draw.line(s, (18, 16, 16), (cx, cy + 1), (cx, cy - 13 * sc), 3)
            pygame.draw.line(s, (18, 16, 16), (cx, cy - 7 * sc), (cx - 5, cy - 11 * sc), 2)
            pygame.draw.line(s, (18, 16, 16), (cx, cy - 9 * sc), (cx + 5, cy - 14 * sc), 2)
            return
        pygame.draw.rect(s, (70, 48, 30), (cx - 1.5, cy - 6 * sc, 3, 8 * sc))
        heat = max(0, (4 - w.fire_dist[y][x]) / 4) * .6
        for i in range(3):
            wid, by = (17 - 4 * i) * sc, cy - (5 + i * 6) * sc
            col = lerp((28 + i * 9, 78 + i * 12, 36 + i * 4), (210, 100, 30), heat)
            if st == BURNING:
                col = lerp(col, (120, 30, 10), .8)
            pygame.draw.polygon(s, col, [(cx, by - 10 * sc), (cx - wid / 2, by), (cx + wid / 2, by)])
        if st == BURNING:
            self.flames(cx, cy - 8 * sc, 20 * sc, x * 7 + y)

    def draw_object(self, x, y, cx, cy):
        w, s = self.w, self.s
        t, st = w.terrain[y][x], w.state[y][x]
        if t == TREE:
            self.draw_tree(x, y, cx, cy, st)
        elif t == ROCK:
            pts = [(-7, 0), (-5, -7), (1, -10), (7, -5), (8, 0), (3, 3), (-4, 3)]
            pygame.draw.polygon(s, (92, 92, 98), [(cx + a, cy + b) for a, b in pts])
            pygame.draw.polygon(s, (130, 130, 138), [(cx + a * .6 - 1, cy + b * .6 - 3) for a, b in pts[1:5]])
        elif t == BRUSH:
            if st == UNBURNT:
                for i in range(4):
                    ox = (i - 1.5) * 5
                    pygame.draw.line(s, (190, 160, 70), (cx + ox, cy + 2),
                                     (cx + ox + (i % 2 * 2 - 1) * 2, cy - 5 - i % 3), 2)
            elif st == BURNING:
                self.flames(cx, cy, 12, x + y * 3)
        elif t == GRASS and st == BURNING:
            self.flames(cx, cy, 8, x * 5 + y)
        elif t == PAD:
            pygame.draw.ellipse(s, (240, 240, 240), (cx - 10, cy - 5, 20, 10), 1)
            self.text("H", cx - 4, cy - 9, (240, 240, 240), self.fs)

    # -- overlay: danger zones, fuses, explored nodes, thermal scan -------------
    def draw_overlay(self, sim):
        w, m, ov = self.w, sim.mission, self.ov
        ov.fill((0, 0, 0, 0))
        pulse = .5 + .5 * math.sin(self.t * 5)
        if self.show_explored:
            for x, y in m.explored[:900]:
                cx, cy = top(w, x, y)
                pygame.draw.polygon(ov, (80, 140, 255, 70), self.diamond(cx, cy))
        for y in range(GRID):
            for x in range(GRID):
                d, t, st = w.fire_dist[y][x], w.terrain[y][x], w.state[y][x]
                if d > 6 and not (t == BRUSH and st == UNBURNT):
                    continue
                cx, cy = top(w, x, y)
                if st != BURNING and 1 <= d <= 3 and t != WATER:
                    a = (95, 65, 40)[d - 1] * (.8 + .2 * pulse)
                    pygame.draw.polygon(ov, (255, 40, 20, int(a)), self.diamond(cx, cy))
                elif w.smoke[y][x] and st != BURNING:
                    pygame.draw.polygon(ov, (150, 150, 170, 40), self.diamond(cx, cy))
                if t == BRUSH and st == UNBURNT:                         # fire fuse marker
                    hot = d <= 6
                    col = (255, 80, 20, int(150 + 100 * pulse)) if hot else (255, 200, 40, 130)
                    pygame.draw.polygon(ov, col, self.diamond(cx, cy, .9), 2)
                    if hot and d <= 5:
                        self.ov.blit(self.fs.render("!", True, (255, 220, 60)), (cx - 2, cy - 7))
        if m.state == "SCAN":                                            # thermal scan ring
            bx, by = m.base
            ring = []
            for i in range(49):
                a = i / 48 * 2 * math.pi
                ring.append((OX + ((bx + m.scan_r * math.cos(a)) - (by + m.scan_r * math.sin(a))) * TW / 2,
                             OY + ((bx + m.scan_r * math.cos(a)) + (by + m.scan_r * math.sin(a))) * TH / 2))
            pygame.draw.lines(ov, (80, 230, 255, 200), False, ring, 3)
        self.s.blit(ov, (0, 0))

    # -- units: victims, drone, path --------------------------------------------
    def person(self, x, y, wave=0.0, col=(255, 140, 30)):
        s = self.s
        pygame.draw.circle(s, (240, 205, 170), (x, y - 11), 3)
        pygame.draw.line(s, col, (x, y - 8), (x, y - 2), 3)
        pygame.draw.line(s, col, (x, y - 7), (x + 5, y - 12 + wave), 2)
        pygame.draw.line(s, col, (x, y - 7), (x - 5, y - 11 - wave), 2)

    def draw_victims(self, m):
        w = self.w
        for v in m.victims:
            if v.status in ("hidden", "carried", "rescued"):
                continue
            cx, cy = top(w, *v.cell)
            if v.status == "lost":
                pygame.draw.line(self.s, (90, 90, 90), (cx - 5, cy - 8), (cx + 5, cy), 2)
                pygame.draw.line(self.s, (90, 90, 90), (cx + 5, cy - 8), (cx - 5, cy), 2)
                continue
            r = 10 + 5 * ((self.t * 1.5) % 1)                             # rescue beacon
            pygame.draw.ellipse(self.s, (80, 230, 255), (cx - r * 1.6, cy - r * .8, r * 3.2, r * 1.6), 1)
            self.person(cx, cy, 3 * math.sin(self.t * 9))
            self.text(f"V{v.idx + 1}", cx - 8, cy - 28, (255, 255, 120), self.fs)

    def draw_drone(self, m):
        w, s = self.w, self.s
        gx, gy = frac_top(w, *m.pos)
        alt = 0 if m.state == "CRASHED" else 30 + 3 * math.sin(self.t * 3)
        dx, dy = gx, gy - alt
        pygame.draw.ellipse(s, (8, 8, 10), (gx - 10, gy - 3, 20, 7))
        if m.state == "RESCUE" and m.target:
            vx, vy = top(w, *m.target.cell)
            pygame.draw.line(s, (230, 230, 230), (dx, dy + 4), (vx, vy - 8), 1)
        if m.carrying:
            pygame.draw.line(s, (230, 230, 230), (dx, dy + 4), (dx, dy + 16), 1)
            self.person(dx, dy + 26, 2 * math.sin(self.t * 7))
        for sx, sy in ((-13, -3), (13, -3), (-9, 5), (9, 5)):
            pygame.draw.line(s, (40, 44, 52), (dx, dy), (dx + sx, dy + sy), 2)
            rx, ry = dx + sx, dy + sy - 2
            a = m.rotor + sx
            pygame.draw.ellipse(s, (120, 130, 150), (rx - 8, ry - 3, 16, 6), 1)
            pygame.draw.line(s, (200, 205, 215), (rx + math.cos(a) * 8, ry + math.sin(a) * 3),
                             (rx - math.cos(a) * 8, ry - math.sin(a) * 3), 1)
        pygame.draw.ellipse(s, (62, 68, 84), (dx - 8, dy - 4, 16, 8))
        pygame.draw.ellipse(s, (170, 185, 210), (dx - 4, dy - 5, 8, 5))
        led = (255, 50, 50) if int(self.t * 4) % 2 else (50, 255, 90)
        pygame.draw.circle(s, led, (int(dx), int(dy + 3)), 2)

    def draw_path(self, m):
        if not self.show_path or m.state in ("SCAN", "DONE", "CRASHED"):
            return
        w = self.w
        pts = [frac_top(w, *m.pos)]
        pts += [top(w, *c) for c in ([m.nxt] if m.nxt else []) + m.path]
        pts = [(x, y - 3) for x, y in pts]
        if len(pts) < 2:
            return
        pygame.draw.lines(self.s, (15, 80, 100), False, pts, 6)
        pygame.draw.lines(self.s, (90, 235, 255), False, pts, 2)
        for x, y in pts[1::2]:
            pygame.draw.circle(self.s, (200, 250, 255), (int(x), int(y)), 2)

    # -- HUD -----------------------------------------------------------------------
    def draw_hud(self, sim, paused, speed):
        s, m, w = self.s, sim.mission, self.w
        s.set_clip(None)
        pygame.draw.rect(s, (18, 20, 27), (MAPW, 0, PANEL, H))
        pygame.draw.line(s, (60, 66, 80), (MAPW, 0), (MAPW, H), 2)
        x, y = MAPW + 16, 14
        self.text("WARD  Wildfire Rescue Drone", x, y, (255, 170, 60), self.fb)
        self.text(f"forest seed {sim.seed}   speed {speed:g}x" + ("  [PAUSED]" if paused else ""),
                  x, y + 26, (150, 156, 170), self.fs)
        y += 52
        st = STATE_TEXT[m.state]
        if m.state in ("TO_VICTIM", "RESCUE") and m.target:
            st = st.replace("victim", f"victim #{m.target.idx + 1}")
        col = (120, 255, 150) if m.state == "DONE" else (255, 90, 80) if m.state in ("CRASHED", "HOLD") else (120, 220, 255)
        self.text(st, x, y, col, self.f)
        y += 22
        self.text(f"mission time {m.clock:6.1f}s", x, y)
        y += 20
        self.text("drone health", x, y)
        pygame.draw.rect(s, (50, 54, 66), (x + 110, y + 2, 150, 12))
        pygame.draw.rect(s, lerp((230, 60, 40), (80, 220, 100), max(0, m.health) / 100),
                         (x + 110, y + 2, 1.5 * max(0, m.health), 12))
        y += 30
        self.text("VICTIMS", x, y, (255, 170, 60))
        y += 20
        colors = {"hidden": (130, 130, 140), "located": (255, 230, 90), "carried": (120, 220, 255),
                  "rescued": (110, 255, 140), "lost": (255, 90, 80)}
        for v in m.victims:
            self.text(f"#{v.idx + 1} {str(v.cell):9s} {v.status}", x, y, colors[v.status])
            y += 18
        y += 12
        self.text(f"A* SEARCH  [{m.heuristic}]", x, y, (255, 170, 60))
        y += 20
        sts = m.stats
        for line in (f"nodes (last)    {sts['last_nodes']}", f"path cost       {sts['last_cost']:.1f}",
                     f"path length     {sts['last_len']} steps", f"search time     {sts['last_ms']:.1f} ms",
                     f"searches        {sts['searches']}", f"total nodes     {sts['total_nodes']}",
                     f"reroutes        {sts['reroutes']}"):
            self.text(line, x, y)
            y += 18
        y += 12
        self.text("ENVIRONMENT", x, y, (255, 170, 60))
        y += 20
        wx, wy = w.wind
        name = {(0, -1): "N-E", (1, -1): "E", (1, 0): "S-E", (1, 1): "S", (0, 1): "S-W",
                (-1, 1): "W", (-1, 0): "N-W", (-1, -1): "N"}[w.wind]
        self.text(f"wind (screen) -> {name}", x, y)
        ax, ay = x + 215, y + 8
        dxs, dys = (wx - wy) * 9, (wx + wy) * 4.5
        pygame.draw.line(s, (200, 220, 255), (ax - dxs, ay - dys), (ax + dxs, ay + dys), 2)
        pygame.draw.circle(s, (200, 220, 255), (int(ax + dxs), int(ay + dys)), 3)
        y += 20
        self.text(f"burning cells   {len(w.burning)}", x, y)
        y += 28
        for line in ("SPACE pause   +/- speed", "H heuristic   E explored", "P path   W wind",
                     "CLICK ignite fire", "R restart   N new forest"):
            self.text(line, x, y, (130, 136, 150), self.fs)
            y += 16
        by = H - 92                                                      # event log
        pygame.draw.rect(s, (10, 11, 16), (0, by - 8, MAPW, 100))
        for i, line in enumerate(m.log[-5:]):
            self.text(line, 14, by + i * 17, (200, 205, 215), self.fs)
        if m.state in ("DONE", "CRASHED"):
            ok = sum(v.status == "rescued" for v in m.victims)
            msg = f"{'MISSION COMPLETE' if m.state == 'DONE' else 'DRONE LOST'}  -  {ok}/{len(m.victims)} rescued"
            tw = self.fb.size(msg)[0]
            pygame.draw.rect(s, (10, 12, 18), (MAPW // 2 - tw // 2 - 16, 40, tw + 32, 36))
            self.text(msg, MAPW // 2 - tw // 2, 48, (120, 255, 150) if ok else (255, 90, 80), self.fb)

    # -- full frame --------------------------------------------------------------------
    def draw(self, sim, paused, speed):
        self.w, w, m = sim.world, sim.world, sim.mission
        self.t = pygame.time.get_ticks() / 1000
        s = self.s
        s.set_clip((0, 0, MAPW, H))
        s.blit(self.bg, (0, 0))
        for y in range(GRID):                                    # terrain + objects, back to front
            for x in range(GRID):
                cx, cy = top(w, x, y)
                col = self.tile_color(x, y)
                dp = w.height[y][x] * HP + 5
                pygame.draw.polygon(s, shade(col, .68), [(cx - TW / 2, cy), (cx, cy + TH / 2),
                                                         (cx, cy + TH / 2 + dp), (cx - TW / 2, cy + dp)])
                pygame.draw.polygon(s, shade(col, .5), [(cx + TW / 2, cy), (cx, cy + TH / 2),
                                                        (cx, cy + TH / 2 + dp), (cx + TW / 2, cy + dp)])
                pygame.draw.polygon(s, col, self.diamond(cx, cy))
                self.draw_object(x, y, cx, cy)
        self.draw_overlay(sim)
        for p in sim.smoke:                                      # smoke
            r = int(p[6] + 9 * p[4] / p[5]) // 2 * 2 + 2
            surf = self.puff(r)
            surf.set_alpha(int(150 * (1 - p[4] / p[5])))
            s.blit(surf, (p[0] - r, p[1] - r))
        dk = self.dark                                           # lighting: darkness minus light sources
        dk.fill((6, 8, 26, 150))
        burning = list(w.burning)
        stride = 1 if len(burning) < 180 else 2
        glows = []
        for i, (x, y) in enumerate(burning):
            if i % stride:
                continue
            cx, cy = top(w, x, y)
            fl = .85 + .15 * math.sin(self.t * 11 + x * 3 + y)
            r = max(12, int(46 * fl * (1.2 if w.terrain[y][x] == TREE else 1)) // 6 * 6)
            dk.blit(self.glow_cut(r), (cx - r, cy - 8 - r), special_flags=pygame.BLEND_RGBA_SUB)
            glows.append((cx - r, cy - 8 - r, r))
        bx, by = top(w, *w.base)
        dk.blit(self.glow_cut(48), (bx - 48, by - 48), special_flags=pygame.BLEND_RGBA_SUB)
        gx, gy = frac_top(w, *m.pos)
        dk.blit(self.glow_cut(54), (gx - 54, gy - 30 - 54), special_flags=pygame.BLEND_RGBA_SUB)
        s.blit(dk, (0, 0))
        for gx0, gy0, r in glows:                                # additive fire glow
            s.blit(self.glow_add(r, (110, 48, 10)), (gx0, gy0), special_flags=pygame.BLEND_RGB_ADD)
        for p in sim.embers:                                     # embers
            f = 1 - p[4] / p[5]
            pygame.draw.circle(s, (255, int(120 + 100 * f), 40), (int(p[0]), int(p[1])), 2 if f > .4 else 1)
        self.draw_victims(m)
        self.draw_path(m)
        self.draw_drone(m)
        self.draw_hud(sim, paused, speed)


# ----------------------------------------------------------------------------
# Entry points
# ----------------------------------------------------------------------------
def run_gui(seed, heuristic):
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("WARD - Wildfire Autonomous Rescue Drone (A*)")
    clock = pygame.time.Clock()
    rend = Renderer(screen)
    sim = Simulation(seed, heuristic)
    paused, speeds, si, running = False, [0.5, 1, 2, 4], 1, True
    while running:
        dt = min(clock.tick(60) / 1000, 0.05)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    si = min(len(speeds) - 1, si + 1)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    si = max(0, si - 1)
                elif k == pygame.K_h:
                    sim.mission.heuristic = HEURISTICS[(HEURISTICS.index(sim.mission.heuristic) + 1) % len(HEURISTICS)]
                elif k == pygame.K_e:
                    rend.show_explored = not rend.show_explored
                elif k == pygame.K_p:
                    rend.show_path = not rend.show_path
                elif k == pygame.K_w:
                    sim.world.wind = DIRS[(DIRS.index(sim.world.wind) + 1) % 8]
                    sim.world.recompute_hazards()
                elif k == pygame.K_r:
                    sim = Simulation(sim.seed, sim.mission.heuristic)
                elif k == pygame.K_n:
                    sim = Simulation(sim.seed + 1, sim.mission.heuristic)
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1 and e.pos[0] < MAPW:
                c = screen_to_cell(sim.world, *e.pos)
                if c and sim.world.ignite(*c):
                    sim.world.recompute_hazards()
                    sim.mission.say(f"Fire ignited at {c}")
        if not paused:
            sim.update(dt * speeds[si])
            sim.update_particles(dt * speeds[si])
        rend.draw(sim, paused, speeds[si])
        pygame.display.flip()
    pygame.quit()


def run_headless(seed, heuristic):
    sim = Simulation(seed, heuristic)
    while sim.mission.state not in ("DONE", "CRASHED") and sim.time < 300:
        sim.update(0.05)
    if sim.mission.state not in ("DONE", "CRASHED"):
        print("Time limit reached")
    return sim


def benchmark(seeds=30, fire_steps=8):
    """Compare heuristics on identical scenarios (base -> each victim, fire already spreading)."""
    agg = {n: dict(found=0, nodes=0, cost=0.0, ms=0.0, ratio=0.0) for n in HEURISTICS}
    runs = 0
    for sd in range(seeds):
        w = make_world(sd)
        for _ in range(fire_steps):
            w.step_fire()
        w.recompute_hazards()
        for v in w.victim_cells:
            res = {n: astar(w, w.base, v, n) for n in HEURISTICS}
            if not res["dijkstra"]["found"]:
                continue
            runs += 1
            opt = res["dijkstra"]["cost"]
            for n, r in res.items():
                if r["found"]:
                    a = agg[n]
                    a["found"] += 1
                    a["nodes"] += r["expanded"]
                    a["cost"] += r["cost"]
                    a["ms"] += r["ms"]
                    a["ratio"] += r["cost"] / opt
    print(f"\nA* heuristic comparison - {runs} searches, {seeds} forests, fire after {fire_steps} steps")
    print(f"{'heuristic':12s}{'found':>7s}{'avg nodes':>11s}{'avg cost':>10s}{'cost/optimal':>14s}{'avg ms':>9s}")
    for n, a in agg.items():
        f = max(1, a["found"])
        print(f"{n:12s}{a['found']:>7d}{a['nodes'] / f:>11.0f}{a['cost'] / f:>10.1f}"
              f"{a['ratio'] / f:>14.3f}{a['ms'] / f:>9.2f}")
    print("(dijkstra = A* with h=0; octile is admissible so its cost equals optimal)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="WARD - wildfire rescue drone (A* search)")
    ap.add_argument("--seed", type=int, default=7, help="forest seed")
    ap.add_argument("--heuristic", choices=HEURISTICS, default="octile")
    ap.add_argument("--headless", action="store_true", help="run one mission without a window")
    ap.add_argument("--benchmark", action="store_true", help="compare heuristics")
    a = ap.parse_args()
    if a.benchmark:
        benchmark()
    elif a.headless:
        run_headless(a.seed, a.heuristic)
    else:
        run_gui(a.seed, a.heuristic)
