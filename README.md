# WARD – Wildfire Autonomous Rescue Drone

**CF321 – Fundamentals of AI · Project 1 (Search Algorithms)**
Innovation University – Faculty of Computers and Information Technology

**Algorithm:** A\* Search  **Language:** Python 3 + pygame

## Team

Individual work
Name: Martina Nasser Kamal

---

## 1. Idea

WARD is an autonomous rescue drone that flies into a burning forest, locates trapped people, and carries them back to a helipad. The fire keeps spreading while the drone flies, so a route that is safe now may be deadly a few seconds later. The drone therefore uses **A\*** to find the cheapest, safest route, and **re-plans after every fire step**.

The forest is shown as a 2.5D isometric scene with:

- Height-mapped terrain, dense trees, boulders and ponds
- Fire spread influenced by wind, slope and flying embers
- **Fire fuses**: strips of dry brush that ignite almost instantly (marked with orange outlines and a "!" when fire is near)
- Danger zones around flames, a smoke plume that drifts downwind
- Day/night lighting where the fire lights up its surroundings, plus smoke and glowing embers
- A thermal scan that locates victims, a winch rescue, and extraction back to base

## 2. Problem Definition

| Element | Definition |
|---------|-----------|
| **State** | The drone's grid cell `(x, y)` together with the current fire map (which cells are burning, burnt or unburnt). The fire map changes over time, so the problem is re-solved from the current state whenever it changes. |
| **Initial state** | Helipad (base) at the start of the mission; later, the drone's current cell. |
| **Goal state** | The cell of the chosen victim (`TO_VICTIM`), then the helipad while carrying them (`TO_BASE`). |
| **Actions** | Move to any of the 8 neighbouring cells (N, S, E, W and the diagonals). Diagonal moves cannot cut corners past a blocked cell. |
| **Blocked cells** | Boulders, burning cells, and fire-fuse cells within 2 cells of a fire. |
| **Action cost** | `step_length × (terrain_cost + risk)` where `step_length` is 1 (straight) or √2 (diagonal). |

### Cost of entering a cell

```
terrain_cost : grass / water / brush / helipad / burnt ground = 1.0, tree canopy = 1.6
risk         : distance to nearest fire  1 -> +14,  2 -> +7,  3 -> +3,  4 -> +1
               smoke (downwind plume)    +3
               live fire fuse near fire  +6
```

Because danger and smoke are part of the path cost, the **cheapest path is also the safest path**. Every cell costs at least 1, so the heuristics below stay meaningful.

## 3. Algorithm: A\* Search

A\* expands the node with the lowest `f(n) = g(n) + h(n)`, where `g` is the cost so far (including risk) and `h` estimates the remaining cost to the goal. It uses a priority queue (`heapq`) and a closed set, and returns the path, its cost and the number of nodes explored.

### Heuristics

| Name | Formula | Admissible? | Purpose |
|------|---------|-------------|---------|
| `octile` (default) | `dx + dy + (√2 − 2)·min(dx, dy)` | Yes | Exact distance on an empty 8-connected grid; never over-estimates because every cell costs ≥ 1, so A\* returns the optimal path. |
| `risk-aware` | `octile × (1 + 0.1 × average risk along the straight line to the goal)` | No | **Creative contribution.** Samples the hazard map between the node and the goal and raises the estimate when the direct line crosses fire, steering the search toward safer corridors. Trades a little optimality for fewer nodes. |
| `manhattan` | `dx + dy` | No | Baseline for comparison (over-estimates diagonal moves). |
| `dijkstra` | `0` | Yes | A\* with no heuristic (uniform-cost search), as a baseline. |

### Dynamic re-planning

The fire advances every 0.5 s of simulated time. After each step the hazard maps (distance to fire, smoke, risk, blocked cells) are recomputed and A\* runs again from the drone's current position to its goal. If no safe path exists, the drone hovers in `HOLD` and retries on the next step.

## 4. Mission Flow

1. **SCAN** – a thermal sweep from the base locates the victims.
2. **PLAN** – A\* is run to every located victim; the cheapest reachable one is chosen.
3. **TO_VICTIM** – fly the path, re-planning as the fire spreads.
4. **RESCUE** – hover and winch the victim aboard (2 s).
5. **TO_BASE** – A\* back to the helipad.
6. **DROP** – land, deliver the victim, and go back to step 2 until no victims remain.

Victims are lost if fire reaches their cell. The drone takes heat damage in or next to burning cells and is lost if its health reaches 0.

## 5. How to Run

Requires Python 3.9+.

```bash
pip install pygame
python wildfire_rescue.py
```

Options:

```bash
python wildfire_rescue.py --seed 12                 # different forest layout
python wildfire_rescue.py --heuristic risk-aware    # octile | risk-aware | manhattan | dijkstra
python wildfire_rescue.py --headless                # run one mission, print results, no window
python wildfire_rescue.py --benchmark               # compare heuristics on many scenarios
```

### Controls

| Key / mouse | Action |
|-------------|--------|
| `SPACE` | Pause / resume |
| `+` / `-` | Change simulation speed (0.5×, 1×, 2×, 4×) |
| `H` | Cycle the heuristic |
| `E` | Show the nodes explored by the last A\* search |
| `P` | Show / hide the planned path |
| `W` | Rotate the wind direction |
| Left click | Ignite a fire at the clicked cell |
| `R` | Restart the same forest |
| `N` | Generate a new forest |
| `ESC` | Quit |

### Reading the scene

- **Cyan line** – the drone's current A\* path
- **Red tint** – danger zone around flames
- **Orange outlined cells** – fire fuses (flashing with "!" when fire is close)
- **Grey-purple tint** – smoke (extra cost)
- **Yellow "V" with cyan ring** – a located victim
- **HUD (right)** – mission state, victims, drone health, and A\* statistics

## 6. Results

The HUD and the console report, for each search: whether a solution was found, the **solution cost**, the **path length**, the **number of nodes explored** and the search time. The totals (searches, total nodes, reroutes) are printed when a mission ends.

`--benchmark` runs all four heuristics on the same scenarios (30 forests, base → each victim, with the fire already spreading). Example output from our test run (20 forests, 59 searches):

| Heuristic | Avg nodes explored | Avg cost | Cost / optimal |
|-----------|-------------------|----------|----------------|
| octile | 238 | 38.2 | 1.000 |
| risk-aware | 213 | 39.2 | 1.020 |
| manhattan | 194 | 38.7 | 1.019 |
| dijkstra (h = 0) | 471 | 38.2 | 1.000 |

`octile` finds the optimal path while exploring about half the nodes of uninformed search. `risk-aware` explores slightly fewer nodes at a small cost in optimality. Run the benchmark yourself to reproduce the numbers; they vary with the seeds.

## 7. Project Structure

```
wildfire_rescue.py   # whole project: world + fire model, A*, mission logic, pygame renderer
README.md
```

Main parts of `wildfire_rescue.py`:

- `World` – terrain generation, fire spread, hazard maps and cell costs
- `astar()` / `make_heuristic()` – the A\* search and its heuristics
- `Mission` – the drone's state machine and re-planning
- `Simulation` / `Renderer` – time stepping, particles and isometric drawing

## 8. Configuration

Constants at the top of `wildfire_rescue.py` can be tweaked: `GRID` (map size), `FIRE_TICK` (fire speed), `DRONE_SPEED`, `NUM_VICTIMS`, `IGNITE_P` (how fast each fuel type catches fire), and `DANGER` / `SMOKE_COST` (the risk penalties). Lower `GRID` or the tile size `TW`/`TH` if the frame rate is low.
