#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSC2103 Data Structures and Algorithms - Group Project (April 2026)
Problem 3: Heuristic Algorithms - A* Search for Path Finding

Console program that finds a lowest-cost route across a weighted grid map.
Supports 4-way and 8-way movement, five heuristics, and a tunable heuristic
weight for trading optimality against speed.

Everything algorithmic is written by hand: the priority queue (MinHeap), the
search loop (astar), the heuristics, and the path reconstruction. No heapq,
no sorted(), no graph or optimisation libraries. The only imports are math
(sqrt), time (timing stats), random (test map generation) and os (file check).

A test and validation suite is built into this same file. Reach it from menu
option 9, or run it directly:

    python3 problem3_astar_pathfinding.py          # interactive menu
    python3 problem3_astar_pathfinding.py test     # run the tests only
"""

import math
import os
import random
import time

# =============================================================================
# TERRAIN AND MAP CONFIGURATION
# =============================================================================
# The grid is a weighted graph. Cost is paid on entering a cell, so a short
# route over mountains can easily be worse than a long route over plains.

WALL_CHAR = '#'

# character -> (terrain name, cost to enter the cell)
TERRAIN = {
    '.': ('Plain',    1),
    '~': ('Water',    3),
    '*': ('Forest',   5),
    '^': ('Mountain', 8),
}

# Characters that may appear in a map file to mark the endpoints.
START_CHAR = 'S'
GOAL_CHAR = 'E'

# Display characters used when drawing a solution.
PATH_CHAR = 'o'      # a cell that belongs to the final path
EXPANDED_CHAR = '+'  # a cell that was expanded (removed from OPEN) by A*

# The 4 orthogonal moves, then the 4 diagonal moves.
# Each entry is (row_delta, col_delta, label).
ORTHOGONAL_MOVES = [
    (-1, 0, 'Up'),
    (1, 0, 'Down'),
    (0, -1, 'Left'),
    (0, 1, 'Right'),
]
DIAGONAL_MOVES = [
    (-1, -1, 'Up-Left'),
    (-1, 1, 'Up-Right'),
    (1, -1, 'Down-Left'),
    (1, 1, 'Down-Right'),
]

SQRT2 = math.sqrt(2.0)

# Floating point tolerance used whenever two costs are compared for equality.
EPS = 1e-9


# =============================================================================
# PRIORITY QUEUE
# =============================================================================
class MinHeap:
    """
    Binary min-heap used as the OPEN list.

    A* repeatedly needs the node with the smallest f. Scanning a list costs
    O(n) each time, so the search ends up O(n^2). A heap makes push and pop
    O(log n) instead.

    Entries are tuples (f, h, seq, row, col). Ordering is by f, then by h
    (prefer nodes nearer the goal, which stops the search drifting sideways
    across ties), then by seq to stay deterministic.

    There is no decrease-key. When a cheaper route to a cell turns up we push
    a new entry and leave the old one behind; the search loop throws these
    stale entries away when they surface. Standard lazy deletion, and it keeps
    the heap simple.
    """

    def __init__(self):
        self._data = []
        self._counter = 0    # insertion sequence, used to break remaining ties
        self.peak_size = 0   # largest size reached, reported in the stats

    @staticmethod
    def _less(a, b):
        # True if a should come out before b. Written out field by field so
        # the tie-break rule is visible and float compares can use EPS.
        if a[0] < b[0] - EPS:
            return True
        if a[0] > b[0] + EPS:
            return False
        if a[1] < b[1] - EPS:
            return True
        if a[1] > b[1] + EPS:
            return False
        return a[2] > b[2]

    def push(self, f, h, row, col):
        entry = (f, h, self._counter, row, col)
        self._counter += 1
        self._data.append(entry)
        self._sift_up(len(self._data) - 1)
        if len(self._data) > self.peak_size:
            self.peak_size = len(self._data)

    def pop(self):
        if not self._data:
            raise IndexError("pop from an empty MinHeap")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last     # move the last item up and push it down
            self._sift_down(0)
        return top

    def is_empty(self):
        return len(self._data) == 0

    def size(self):
        return len(self._data)

    def _sift_up(self, index):
        item = self._data[index]
        while index > 0:
            parent_index = (index - 1) // 2
            parent = self._data[parent_index]
            if self._less(item, parent):
                self._data[index] = parent     # pull the parent down
                index = parent_index
            else:
                break
        self._data[index] = item

    def _sift_down(self, index):
        size = len(self._data)
        item = self._data[index]
        while True:
            left = 2 * index + 1
            right = left + 1
            smallest = index
            candidate = item
            if left < size and self._less(self._data[left], candidate):
                smallest = left
                candidate = self._data[left]
            if right < size and self._less(self._data[right], candidate):
                smallest = right
                candidate = self._data[right]
            if smallest == index:
                break
            self._data[index] = candidate      # pull the smaller child up
            index = smallest
        self._data[index] = item


# =============================================================================
# THE GRID
# =============================================================================
class Grid:
    """A rectangular weighted grid map with obstacles."""

    def __init__(self, rows, name="Untitled map"):
        self.name = name
        self.cells = [list(r) for r in rows]
        self.height = len(self.cells)
        self.width = len(self.cells[0]) if self.height > 0 else 0
        self.start = None
        self.goal = None

        # Pull out the S / E markers and replace them with plain ground, so
        # the cost model does not have to special-case the endpoints.
        for r in range(self.height):
            for c in range(self.width):
                ch = self.cells[r][c]
                if ch == START_CHAR:
                    self.start = (r, c)
                    self.cells[r][c] = '.'
                elif ch == GOAL_CHAR:
                    self.goal = (r, c)
                    self.cells[r][c] = '.'

    @staticmethod
    def validate_rows(rows):
        """
        Check raw map lines. Returns an error message, or None if the map is
        fine. Every input path calls this so bad data is caught before the
        search starts rather than crashing halfway through.
        """
        if not rows:
            return "The map is empty."
        width = len(rows[0])
        if width == 0:
            return "The first row of the map is empty."
        allowed = set(TERRAIN.keys())
        allowed.add(WALL_CHAR)
        allowed.add(START_CHAR)
        allowed.add(GOAL_CHAR)
        starts = 0
        goals = 0
        for i, row in enumerate(rows):
            if len(row) != width:
                return ("Row %d has %d characters but row 0 has %d. "
                        "All rows must be the same length."
                        % (i, len(row), width))
            for ch in row:
                if ch not in allowed:
                    return ("Row %d contains the unknown character '%s'. "
                            "Allowed characters: %s"
                            % (i, ch, " ".join(sorted_chars(allowed))))
            starts += row.count(START_CHAR)
            goals += row.count(GOAL_CHAR)
        if starts > 1:
            return "The map contains more than one 'S' (start) marker."
        if goals > 1:
            return "The map contains more than one 'E' (end) marker."
        return None

    def in_bounds(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def is_wall(self, r, c):
        return self.cells[r][c] == WALL_CHAR

    def is_passable(self, r, c):
        return self.in_bounds(r, c) and not self.is_wall(r, c)

    def enter_cost(self, r, c):
        return TERRAIN[self.cells[r][c]][1]

    def terrain_name(self, r, c):
        if self.is_wall(r, c):
            return "Wall"
        return TERRAIN[self.cells[r][c]][0]

    def min_terrain_cost(self):
        # Cheapest passable cell in this map. The heuristic is scaled by it so
        # it can never claim more than the agent could actually be charged.
        best = None
        for r in range(self.height):
            for c in range(self.width):
                if not self.is_wall(r, c):
                    cost = self.enter_cost(r, c)
                    if best is None or cost < best:
                        best = cost
        return 1 if best is None else best

    def passable_count(self):
        total = 0
        for r in range(self.height):
            for c in range(self.width):
                if not self.is_wall(r, c):
                    total += 1
        return total

    def neighbours(self, r, c, allow_diagonal, allow_corner_cutting):
        """
        Yield (nr, nc, step_cost, label) for each legal move out of (r, c).

        Diagonals cost sqrt(2) x terrain so the geometry stays consistent.
        With corner cutting off, a diagonal needs both orthogonal neighbours
        clear, otherwise the agent could slip through the join between two
        walls that only touch at a point.
        """
        moves = ORTHOGONAL_MOVES if not allow_diagonal else \
            ORTHOGONAL_MOVES + DIAGONAL_MOVES

        for dr, dc, label in moves:
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc):
                continue
            if self.is_wall(nr, nc):
                continue
            if dr != 0 and dc != 0 and not allow_corner_cutting:
                if self.is_wall(r + dr, c) or self.is_wall(r, c + dc):
                    continue
            step = self.enter_cost(nr, nc)
            if dr != 0 and dc != 0:
                step *= SQRT2
            yield nr, nc, step, label


def sorted_chars(char_set):
    # Insertion sort. Only used to tidy up error messages, but built-in
    # sorting is off limits for this assignment so it is written out.
    items = list(char_set)
    for i in range(1, len(items)):
        key = items[i]
        j = i - 1
        while j >= 0 and items[j] > key:
            items[j + 1] = items[j]
            j -= 1
        items[j + 1] = key
    return items


# =============================================================================
# HEURISTICS
# =============================================================================
# h(n) estimates the cost left to reach the goal. A* only guarantees an
# optimal path when h never overestimates (admissible), so each formula is
# scaled by the cheapest terrain in the map.

HEURISTICS = ['manhattan', 'euclidean', 'octile', 'chebyshev', 'zero']

HEURISTIC_NOTES = {
    'manhattan': "|dr| + |dc|  - exact for 4-way movement; OVERESTIMATES when "
                 "diagonal movement is enabled (not admissible there).",
    'euclidean': "sqrt(dr^2 + dc^2) - straight line distance; always "
                 "admissible but weak (loose) for 4-way movement.",
    'octile':    "diagonal-aware distance - exact for 8-way movement with "
                 "diagonal cost sqrt(2); the correct choice for 8-way.",
    'chebyshev': "max(|dr|, |dc|) - admissible but very loose; assumes a "
                 "diagonal move costs the same as an orthogonal one.",
    'zero':      "h(n) = 0 - A* degenerates into Dijkstra / uniform-cost "
                 "search; still optimal but explores far more nodes.",
}


def heuristic(name, r, c, goal_r, goal_c, scale):
    dr = abs(r - goal_r)
    dc = abs(c - goal_c)

    if name == 'manhattan':
        return (dr + dc) * scale
    if name == 'euclidean':
        return math.sqrt(dr * dr + dc * dc) * scale
    if name == 'octile':
        # diagonal for min(dr, dc) steps, then straight for what is left
        smaller = dr if dr < dc else dc
        larger = dr if dr > dc else dc
        return ((larger - smaller) + SQRT2 * smaller) * scale
    if name == 'chebyshev':
        return (dr if dr > dc else dc) * scale
    if name == 'zero':
        return 0.0
    raise ValueError("Unknown heuristic: %s" % name)


def heuristic_is_admissible(name, allow_diagonal):
    # Manhattan double-counts under 8-way movement: one diagonal step cuts
    # both dr and dc, but Manhattan charges for each separately. Everything
    # else here is safe.
    if name == 'manhattan' and allow_diagonal:
        return False
    return True


# =============================================================================
# A* SEARCH
# =============================================================================
def astar(grid, start, goal,
          heuristic_name='manhattan',
          allow_diagonal=False,
          allow_corner_cutting=False,
          weight=1.0,
          record_expanded=True):
    """
    Best-first search driven by f(n) = g(n) + w * h(n), where g is the cost
    already paid and h is the estimate of what is left. Always expands the
    smallest f, so the heuristic pulls the search towards the goal instead of
    letting it spread out evenly like Dijkstra does.

    w = 1 is ordinary A*. Larger w makes it greedier: faster, but the path
    may cost more than the optimum.

    Returns a dict holding the path and the search statistics.
    """
    result = {
        'found': False,
        'path': [],
        'cost': 0.0,
        'g_values': [],
        'h_values': [],
        'labels': [],
        'step_costs': [],
        'expanded': 0,
        'generated': 0,
        'peak_open': 0,
        'stale_skipped': 0,
        'elapsed_ms': 0.0,
        'expanded_cells': set(),
        'heuristic': heuristic_name,
        'weight': weight,
        'allow_diagonal': allow_diagonal,
        'admissible': heuristic_is_admissible(heuristic_name, allow_diagonal)
                      and abs(weight - 1.0) < EPS,
    }

    start_time = time.perf_counter()

    sr, sc = start
    gr, gc = goal
    scale = grid.min_terrain_cost()

    open_list = MinHeap()          # discovered but not yet expanded

    # g_score : cheapest known cost from the start to each cell
    # parent  : where each cell was reached from, used to rebuild the path
    # closed  : cells already expanded, their g is final
    g_score = {(sr, sc): 0.0}
    parent = {}
    closed = {}

    h_start = heuristic(heuristic_name, sr, sc, gr, gc, scale)
    open_list.push(0.0 + weight * h_start, h_start, sr, sc)

    while not open_list.is_empty():
        f_current, h_current, _seq, r, c = open_list.pop()

        if (r, c) in closed:
            result['stale_skipped'] += 1     # leftover from a superseded push
            continue

        closed[(r, c)] = True
        result['expanded'] += 1
        if record_expanded:
            result['expanded_cells'].add((r, c))

        # Test the goal here, on expansion, not when it is first generated.
        # Accepting it early would mean accepting it while cheaper nodes are
        # still queued, which is exactly how the optimality guarantee is lost.
        if r == gr and c == gc:
            result['found'] = True
            result['cost'] = g_score[(r, c)]
            _reconstruct(result, parent, start, goal,
                         heuristic_name, gr, gc, scale, g_score)
            break

        current_g = g_score[(r, c)]

        for nr, nc, step_cost, label in grid.neighbours(
                r, c, allow_diagonal, allow_corner_cutting):
            if (nr, nc) in closed:
                continue
            tentative_g = current_g + step_cost
            known_g = g_score.get((nr, nc))
            if known_g is None or tentative_g < known_g - EPS:
                # strictly cheaper route to this neighbour, so record it
                g_score[(nr, nc)] = tentative_g
                parent[(nr, nc)] = ((r, c), label, step_cost)
                h_value = heuristic(heuristic_name, nr, nc, gr, gc, scale)
                open_list.push(tentative_g + weight * h_value,
                               h_value, nr, nc)
                result['generated'] += 1

    result['peak_open'] = open_list.peak_size
    result['elapsed_ms'] = (time.perf_counter() - start_time) * 1000.0
    return result


def _reconstruct(result, parent, start, goal,
                 heuristic_name, gr, gc, scale, g_score):
    # Follow the parent chain back from the goal, then flip it. Also collects
    # the per-step figures the output table needs.
    path = []
    labels = []
    steps = []
    node = goal
    while node != start:
        path.append(node)
        prev, label, step_cost = parent[node]
        labels.append(label)
        steps.append(step_cost)
        node = prev
    path.append(start)
    labels.append('-')          # nothing was moved to reach the start
    steps.append(0.0)

    # reverse in place with two pointers
    left, right = 0, len(path) - 1
    while left < right:
        path[left], path[right] = path[right], path[left]
        labels[left], labels[right] = labels[right], labels[left]
        steps[left], steps[right] = steps[right], steps[left]
        left += 1
        right -= 1

    result['path'] = path
    result['labels'] = labels
    result['step_costs'] = steps
    result['g_values'] = [g_score[cell] for cell in path]
    result['h_values'] = [heuristic(heuristic_name, cell[0], cell[1],
                                    gr, gc, scale) for cell in path]


# =============================================================================
# REACHABILITY CHECK (random map generation only)
# =============================================================================
def flood_fill_reachable(grid, start):
    # Plain BFS flood fill, 4-way. Used only to check a generated random map
    # actually has a solution before it is handed to the user. This is not
    # the solution to the problem - astar() above is.
    seen = {start}
    queue = [start]
    head = 0                      # index instead of pop(0), which is O(n)
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc, _label in ORTHOGONAL_MOVES:
            nr, nc = r + dr, c + dc
            if grid.in_bounds(nr, nc) and not grid.is_wall(nr, nc) \
                    and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))
    return seen


# =============================================================================
# OUTPUT FORMATTING
# =============================================================================
LINE = "=" * 78
THIN = "-" * 78


def banner(title):
    print()
    print(LINE)
    print(" " + title)
    print(LINE)


def format_cost(value):
    # keep whole numbers free of a trailing .00 so the tables stay readable
    if abs(value - round(value)) < 1e-6:
        return "%d" % int(round(value))
    return "%.2f" % value


def render_grid(grid, start=None, goal=None,
                path_cells=None, expanded_cells=None, title=None):
    # Draws the map with rulers. What gets shown in a cell, in order of
    # priority: S, G, path, expanded, then the terrain underneath.
    if title:
        print()
        print(title)
    path_cells = path_cells or set()
    expanded_cells = expanded_cells or set()

    # column ruler: tens digits on one line, units on the next
    if grid.width > 10:
        tens = "     "
        for c in range(grid.width):
            tens += ("%d" % (c // 10) if c >= 10 else " ") + " "
        print(tens)
    units = "     "
    for c in range(grid.width):
        units += "%d " % (c % 10)
    print(units)
    print("    +" + "-" * (2 * grid.width - 1) + "+")

    for r in range(grid.height):
        line = "%3d |" % r
        for c in range(grid.width):
            if start is not None and (r, c) == start:
                ch = START_CHAR
            elif goal is not None and (r, c) == goal:
                ch = GOAL_CHAR
            elif (r, c) in path_cells:
                ch = PATH_CHAR
            elif (r, c) in expanded_cells:
                ch = EXPANDED_CHAR
            else:
                ch = grid.cells[r][c]
            line += ch + (" " if c < grid.width - 1 else "")
        line += "|"
        print(line)
    print("    +" + "-" * (2 * grid.width - 1) + "+")


def print_legend():
    banner("MAP LEGEND")
    print("  Terrain (the number is the cost of stepping INTO the cell):")
    for ch in ['.', '~', '*', '^']:
        name, cost = TERRAIN[ch]
        print("     %s  %-9s cost %d" % (ch, name, cost))
    print("     %s  %-9s impassable" % (WALL_CHAR, 'Wall'))
    print()
    print("  Markers:")
    print("     %s  Start cell" % START_CHAR)
    print("     %s  Goal / End cell" % GOAL_CHAR)
    print("     %s  Cell on the final path" % PATH_CHAR)
    print("     %s  Cell expanded by A* but not on the final path" % EXPANDED_CHAR)
    print()
    print("  Diagonal moves cost sqrt(2) x terrain cost when 8-way movement")
    print("  is enabled, so that distances stay geometrically consistent.")


def print_path_table(grid, result, max_rows=24):
    # Step table showing g, h and f at every cell of the path. Long paths
    # get truncated in the middle, otherwise the console fills up.
    path = result['path']
    print()
    print("  STEP-BY-STEP PATH  (f = g + %s x h)"
          % format_cost(result['weight']))
    print("  " + THIN)
    print("  %-5s %-9s %-11s %-11s %8s %8s %8s %8s"
          % ("Step", "Cell", "Terrain", "Move", "Step", "g(n)", "h(n)", "f(n)"))
    print("  " + THIN)

    total = len(path)
    if total <= max_rows:
        show = list(range(total))
    else:
        half = max_rows // 2
        show = list(range(half)) + [-1] + list(range(total - half, total))

    for i in show:
        if i == -1:
            print("  %-5s %-9s %-11s %-11s %8s %8s %8s %8s"
                  % ("...", "...", "...", "...", "...", "...", "...", "..."))
            continue
        r, c = path[i]
        g = result['g_values'][i]
        h = result['h_values'][i]
        f = g + result['weight'] * h
        print("  %-5d %-9s %-11s %-11s %8s %8s %8s %8s"
              % (i,
                 "(%d,%d)" % (r, c),
                 grid.terrain_name(r, c),
                 result['labels'][i],
                 format_cost(result['step_costs'][i]),
                 format_cost(g),
                 format_cost(h),
                 format_cost(f)))
    print("  " + THIN)
    if total > max_rows:
        print("  (%d intermediate steps hidden for readability)"
              % (total - max_rows))


def print_statistics(grid, result):
    print()
    print("  SEARCH STATISTICS")
    print("  " + THIN)
    print("  %-34s : %s" % ("Heuristic used", result['heuristic']))
    print("  %-34s : %s" % ("Heuristic weight (w)",
                            format_cost(result['weight'])))
    print("  %-34s : %s" % ("Movement rule",
                            "8-way (with diagonals)" if result['allow_diagonal']
                            else "4-way (orthogonal only)"))
    if result['found']:
        print("  %-34s : %s" % ("Total path cost", format_cost(result['cost'])))
        print("  %-34s : %d cells (%d moves)"
              % ("Path length", len(result['path']), len(result['path']) - 1))
    else:
        print("  %-34s : %s" % ("Total path cost", "n/a - no path exists"))
        print("  %-34s : %s" % ("Path length", "n/a - no path exists"))
    print("  %-34s : %d" % ("Nodes expanded (closed)", result['expanded']))
    print("  %-34s : %d" % ("Nodes generated (pushed)", result['generated']))
    print("  %-34s : %d" % ("Peak OPEN list size", result['peak_open']))
    print("  %-34s : %d" % ("Stale heap entries skipped",
                            result['stale_skipped']))
    print("  %-34s : %d" % ("Passable cells in map", grid.passable_count()))
    print("  %-34s : %.1f %%"
          % ("Search space explored",
             100.0 * result['expanded'] / max(1, grid.passable_count())))
    print("  %-34s : %.3f ms" % ("Runtime", result['elapsed_ms']))
    print("  " + THIN)
    if not result['found']:
        print("  Completeness : A* is COMPLETE on a finite grid. Because the")
        print("                 whole reachable region was expanded without")
        print("                 reaching the goal, 'no path exists' is a")
        print("                 proven result, not a failure of the search.")
        return
    if result['admissible']:
        print("  Optimality : GUARANTEED - the heuristic never overestimates")
        print("               the remaining cost (it is admissible) and the")
        print("               weight is 1.0, so this path is a cheapest path.")
    else:
        reason = []
        if not heuristic_is_admissible(result['heuristic'],
                                       result['allow_diagonal']):
            reason.append("the heuristic can overestimate the remaining cost")
        if abs(result['weight'] - 1.0) >= EPS:
            reason.append("the heuristic weight w = %s is greater than 1"
                          % format_cost(result['weight']))
        print("  Optimality : NOT GUARANTEED - " + "; ".join(reason) + ".")
        print("               The path returned is a GOOD path found quickly,")
        print("               but it may not be the cheapest one. This is the")
        print("               usual heuristic trade-off: speed for accuracy.")


# =============================================================================
# BUILT-IN SAMPLE MAPS
# =============================================================================
BUILTIN_MAPS = {
    '1': ("Warehouse floor (obstacles only, 9 x 12)", [
        "S..........#",
        ".####.####.#",
        ".#..#.#..#..",
        ".#.##.#.##.#",
        ".#....#..#..",
        ".#.####..#.#",
        ".#....#..#..",
        ".####.####..",
        "...........E",
    ]),
    '2': ("Terrain map (mixed movement costs, 10 x 14)", [
        "S..~~~........",
        "..~~***~~.....",
        ".~~*^^^*~~....",
        ".~*^^^^^*~~...",
        ".~*^^^^^*~~..E",
        ".~~*^^^*~~....",
        "..~~***~~.....",
        "...~~~........",
        "..............",
        "..............",
    ]),
    '3': ("Zig-zag corridors (worst case for the heuristic, 11 x 15)", [
        "S..............",
        "#############..",
        "...............",
        "..#############",
        "...............",
        "#############..",
        "...............",
        "..#############",
        "...............",
        "#############..",
        "E..............",
    ]),
    '4': ("Open field with a river crossing (8 x 16)", [
        "S...~~..........",
        "....~~..........",
        "....~~....####..",
        "....~~....#..#..",
        "..###~~~..#..#..",
        "....~~....####..",
        "....~~.........E",
        "....~~..........",
    ]),
    '5': ("Sealed room - NO SOLUTION EXISTS (8 x 12)", [
        "S...........",
        "............",
        "...######...",
        "...#....#...",
        "...#..E.#...",
        "...######...",
        "............",
        "............",
    ]),
}


# =============================================================================
# INPUT HELPERS
# =============================================================================
def prompt(message):
    # wrapper so piping input into the program does not blow up on EOF
    try:
        return input(message)
    except EOFError:
        print("\n[Input stream closed - exiting.]")
        raise SystemExit(0)


def ask_int(message, low, high, default=None):
    # keeps asking until the answer is a number inside the range
    while True:
        suffix = "" if default is None else " [default %d]" % default
        raw = prompt("%s (%d-%d)%s: " % (message, low, high, suffix)).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("  ! Please type a whole number.")
            continue
        if value < low or value > high:
            print("  ! Please enter a value between %d and %d." % (low, high))
            continue
        return value


def ask_float(message, low, high, default=None):
    while True:
        suffix = "" if default is None else " [default %s]" % format_cost(default)
        raw = prompt("%s (%s-%s)%s: "
                     % (message, format_cost(low), format_cost(high),
                        suffix)).strip()
        if raw == "" and default is not None:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("  ! Please type a number.")
            continue
        if value < low - EPS or value > high + EPS:
            print("  ! Please enter a value between %s and %s."
                  % (format_cost(low), format_cost(high)))
            continue
        return value


def ask_yes_no(message, default=True):
    hint = "Y/n" if default else "y/N"
    while True:
        raw = prompt("%s (%s): " % (message, hint)).strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  ! Please answer y or n.")


def ask_cell(grid, message):
    # reads a coordinate and rejects anything off the map or inside a wall
    while True:
        row = ask_int("  %s - row" % message, 0, grid.height - 1)
        col = ask_int("  %s - column" % message, 0, grid.width - 1)
        if grid.is_wall(row, col):
            print("  ! Cell (%d,%d) is a wall. Choose a passable cell."
                  % (row, col))
            continue
        return (row, col)


# =============================================================================
# MAP LOADING
# =============================================================================
def choose_builtin_map():
    banner("BUILT-IN SAMPLE MAPS")
    for key in ['1', '2', '3', '4', '5']:
        print("  %s. %s" % (key, BUILTIN_MAPS[key][0]))
    print("  0. Cancel")
    while True:
        choice = prompt("  Select a map: ").strip()
        if choice == '0':
            return None
        if choice in BUILTIN_MAPS:
            name, rows = BUILTIN_MAPS[choice]
            error = Grid.validate_rows(rows)
            if error:                      # defensive: should never happen
                print("  ! Built-in map is invalid: %s" % error)
                return None
            grid = Grid(rows, name)
            print("  Loaded: %s" % name)
            return grid
        print("  ! Invalid selection.")


def load_map_from_file():
    banner("LOAD MAP FROM FILE")
    print("  The file must contain one row of the map per line, using the")
    print("  characters . ~ * ^ # and optionally S (start) and E (end).")
    path = prompt("  File path (or blank to cancel): ").strip()
    if path == "":
        return None
    if not os.path.isfile(path):
        print("  ! File not found: %s" % path)
        return None
    try:
        with open(path, 'r') as handle:
            raw_lines = handle.read().split("\n")
    except OSError as exc:
        print("  ! Could not read the file: %s" % exc)
        return None

    rows = []
    for line in raw_lines:
        stripped = line.rstrip("\r\n")
        if stripped.strip() == "":
            continue                      # skip blank lines
        rows.append(stripped)

    error = Grid.validate_rows(rows)
    if error:
        print("  ! Invalid map file: %s" % error)
        return None
    grid = Grid(rows, os.path.basename(path))
    print("  Loaded '%s'  (%d rows x %d columns)"
          % (grid.name, grid.height, grid.width))
    return grid


def enter_map_manually():
    banner("ENTER A MAP MANUALLY")
    print("  Allowed characters: . (plain 1)  ~ (water 3)  * (forest 5)")
    print("                      ^ (mountain 8)  # (wall)  S (start)  E (end)")
    height = ask_int("  Number of rows", 2, 40)
    width = ask_int("  Number of columns", 2, 40)
    rows = []
    for r in range(height):
        while True:
            line = prompt("  Row %2d (%d characters): " % (r, width)).strip()
            if len(line) != width:
                print("  ! That row has %d characters, %d were expected."
                      % (len(line), width))
                continue
            rows.append(line)
            break
    error = Grid.validate_rows(rows)
    if error:
        print("  ! Invalid map: %s" % error)
        return None
    grid = Grid(rows, "Manually entered map")
    print("  Map accepted.")
    return grid


def generate_random_map():
    banner("GENERATE A RANDOM TEST MAP")
    height = ask_int("  Number of rows", 5, 40, 15)
    width = ask_int("  Number of columns", 5, 40, 25)
    wall_pct = ask_int("  Wall density in percent", 0, 45, 25)
    terrain_pct = ask_int("  Rough terrain density in percent", 0, 60, 30)
    seed_choice = prompt("  Random seed (blank = time based): ").strip()
    if seed_choice != "":
        try:
            random.seed(int(seed_choice))
        except ValueError:
            random.seed(seed_choice)
    else:
        random.seed()

    rough_chars = ['~', '*', '^']

    # Try a number of times to produce a map where the goal is reachable.
    for attempt in range(60):
        rows = []
        for r in range(height):
            row = ""
            for c in range(width):
                roll = random.randint(1, 100)
                if roll <= wall_pct:
                    row += WALL_CHAR
                elif roll <= wall_pct + terrain_pct:
                    row += rough_chars[random.randint(0, 2)]
                else:
                    row += '.'
            rows.append(row)
        grid = Grid(rows, "Random map (%dx%d, %d%% walls)"
                    % (height, width, wall_pct))
        grid.cells[0][0] = '.'
        grid.cells[height - 1][width - 1] = '.'
        start = (0, 0)
        goal = (height - 1, width - 1)
        if goal in flood_fill_reachable(grid, start):
            grid.start = start
            grid.goal = goal
            print("  Generated a solvable map after %d attempt(s)."
                  % (attempt + 1))
            return grid

    print("  ! Could not generate a solvable map with that wall density.")
    print("    Try lowering the wall percentage.")
    return None


# =============================================================================
# RUN AND COMPARE ACTIONS
# =============================================================================
class Settings:
    # current search configuration, edited through menu option 6

    def __init__(self):
        self.heuristic = 'manhattan'
        self.allow_diagonal = False
        self.allow_corner_cutting = False
        self.weight = 1.0
        self.show_footprint = True

    def describe(self):
        return ("heuristic=%s, movement=%s, corner-cutting=%s, w=%s"
                % (self.heuristic,
                   "8-way" if self.allow_diagonal else "4-way",
                   "on" if self.allow_corner_cutting else "off",
                   format_cost(self.weight)))


def configure_settings(settings):
    banner("SEARCH SETTINGS")
    print("  Current: %s" % settings.describe())
    print()
    print("  Available heuristics:")
    for i, name in enumerate(HEURISTICS, start=1):
        print("    %d. %-10s %s" % (i, name, HEURISTIC_NOTES[name]))
    index = ask_int("  Choose a heuristic", 1, len(HEURISTICS),
                    HEURISTICS.index(settings.heuristic) + 1)
    settings.heuristic = HEURISTICS[index - 1]

    settings.allow_diagonal = ask_yes_no(
        "  Allow diagonal (8-way) movement?", settings.allow_diagonal)
    if settings.allow_diagonal:
        settings.allow_corner_cutting = ask_yes_no(
            "  Allow cutting the corner of a wall diagonally?",
            settings.allow_corner_cutting)
        if settings.heuristic == 'manhattan':
            print()
            print("  ! WARNING: Manhattan distance overestimates the remaining")
            print("    cost when diagonal movement is allowed, so it is NOT")
            print("    admissible here and A* may return a non-optimal path.")
            print("    'octile' is the correct heuristic for 8-way movement.")

    print()
    print("  The heuristic weight w in f = g + w*h controls the trade-off:")
    print("    w = 1.0  -> standard A*, optimal path guaranteed")
    print("    w > 1.0  -> weighted A*, faster but possibly non-optimal")
    settings.weight = ask_float("  Heuristic weight w", 1.0, 10.0,
                                settings.weight)

    settings.show_footprint = ask_yes_no(
        "  Draw the search footprint (expanded cells) after each run?",
        settings.show_footprint)
    print()
    print("  Settings updated: %s" % settings.describe())


def choose_endpoints(grid):
    # use the S/E markers from the map, or let the user type coordinates
    if grid.start is not None and grid.goal is not None:
        print("  Map defines start %s and goal %s."
              % (str(grid.start), str(grid.goal)))
        if not ask_yes_no("  Use these cells?", True):
            grid.start = ask_cell(grid, "Start")
            grid.goal = ask_cell(grid, "Goal")
    else:
        print("  The map does not define both endpoints; please enter them.")
        grid.start = ask_cell(grid, "Start")
        grid.goal = ask_cell(grid, "Goal")

    if grid.start == grid.goal:
        print("  ! Start and goal are the same cell.")
        return False
    return True


def run_search(grid, settings):
    banner("A* SEARCH")
    if not choose_endpoints(grid):
        return

    result = astar(grid, grid.start, grid.goal,
                   heuristic_name=settings.heuristic,
                   allow_diagonal=settings.allow_diagonal,
                   allow_corner_cutting=settings.allow_corner_cutting,
                   weight=settings.weight)

    if not result['found']:
        print()
        print("  RESULT: NO PATH EXISTS between %s and %s."
              % (str(grid.start), str(grid.goal)))
        print("  A* expanded every reachable cell (%d) and the goal was never"
              % result['expanded'])
        print("  reached, which proves the goal is enclosed by walls.")
        print_statistics(grid, result)
        return

    print()
    print("  RESULT: PATH FOUND")
    render_grid(grid, grid.start, grid.goal,
                path_cells=set(result['path']),
                title="  FINAL PATH  ('%s' marks the route)" % PATH_CHAR)

    if settings.show_footprint:
        render_grid(grid, grid.start, grid.goal,
                    path_cells=set(result['path']),
                    expanded_cells=result['expanded_cells'],
                    title="  SEARCH FOOTPRINT  ('%s' = expanded but unused)"
                          % EXPANDED_CHAR)

    print_path_table(grid, result)
    print_statistics(grid, result)


def compare_algorithms(grid, settings):
    # Runs several configurations over the same map and tabulates them.
    # Easiest way to show what the heuristic actually buys, and that a
    # heuristic search will not always hand back the optimum.
    banner("COMPARISON: HOW THE HEURISTIC CHANGES THE SEARCH")
    if not choose_endpoints(grid):
        return

    configurations = [
        ("Dijkstra (A* with h = 0)", 'zero', 1.0),
        ("A* + Manhattan", 'manhattan', 1.0),
        ("A* + Euclidean", 'euclidean', 1.0),
        ("A* + Chebyshev", 'chebyshev', 1.0),
        ("Weighted A* (w = 2)", 'manhattan', 2.0),
        ("Weighted A* (w = 5)", 'manhattan', 5.0),
    ]
    if settings.allow_diagonal:
        configurations.insert(2, ("A* + Octile", 'octile', 1.0))

    print()
    print("  Map        : %s (%d x %d)" % (grid.name, grid.height, grid.width))
    print("  Start/Goal : %s -> %s" % (str(grid.start), str(grid.goal)))
    print("  Movement   : %s"
          % ("8-way" if settings.allow_diagonal else "4-way"))
    print()
    print("  %-26s %10s %8s %10s %9s %9s"
          % ("Configuration", "Path cost", "Moves", "Expanded", "Time ms",
             "Optimal?"))
    print("  " + THIN)

    optimal_cost = None
    rows_to_print = []
    for label, h_name, weight in configurations:
        outcome = astar(grid, grid.start, grid.goal,
                        heuristic_name=h_name,
                        allow_diagonal=settings.allow_diagonal,
                        allow_corner_cutting=settings.allow_corner_cutting,
                        weight=weight,
                        record_expanded=False)
        if not outcome['found']:
            rows_to_print.append((label, outcome, None))
            continue
        # Dijkstra goes first and is always exact, so its cost becomes the
        # yardstick for everything below it.
        if optimal_cost is None:
            optimal_cost = outcome['cost']
        rows_to_print.append((label, outcome, optimal_cost))

    for label, outcome, reference in rows_to_print:
        if not outcome['found']:
            print("  %-26s %10s %8s %10d %9.3f %9s"
                  % (label, "no path", "-", outcome['expanded'],
                     outcome['elapsed_ms'], "-"))
            continue
        is_optimal = reference is not None and \
            abs(outcome['cost'] - reference) < 1e-6
        # trailing '*' means no optimality guarantee
        shown = label if outcome['admissible'] else label + " *"
        print("  %-26s %10s %8d %10d %9.3f %9s"
              % (shown,
                 format_cost(outcome['cost']),
                 len(outcome['path']) - 1,
                 outcome['expanded'],
                 outcome['elapsed_ms'],
                 "yes" if is_optimal else "NO"))
    print("  " + THIN)
    print("  * = no optimality guarantee (inadmissible heuristic or w > 1).")
    print("      A 'yes' on such a row means it happened to find the optimum")
    print("      on THIS map, not that it always will.")

    # observations worked out from the numbers just produced
    print("  OBSERVATIONS (calculated from the run above)")
    if optimal_cost is None:
        print("   * The goal is unreachable, so every configuration had to")
        print("     exhaust the whole reachable region before reporting")
        print("     failure. No heuristic can help in this case.")
        return

    dijkstra_expanded = rows_to_print[0][1]['expanded']
    best_label, best_expanded = None, None      # best GUARANTEED-exact run
    fastest_label, fastest_expanded = None, None  # fewest expansions overall
    worst_cost_label, worst_cost = None, optimal_cost
    for label, outcome, _reference in rows_to_print:
        if not outcome['found']:
            continue
        # Only unweighted, admissible runs may be described as guaranteed.
        if outcome['admissible'] and label != rows_to_print[0][0]:
            if best_expanded is None or outcome['expanded'] < best_expanded:
                best_expanded = outcome['expanded']
                best_label = label
        if fastest_expanded is None or outcome['expanded'] < fastest_expanded:
            fastest_expanded = outcome['expanded']
            fastest_label = label
        if outcome['cost'] > worst_cost + 1e-6:
            worst_cost = outcome['cost']
            worst_cost_label = label

    print("   * Optimal path cost on this map = %s, established by Dijkstra"
          % format_cost(optimal_cost))
    print("     (A* with h = 0), which is always exact.")
    if best_label is not None:
        print("   * Best GUARANTEED-optimal configuration: %s," % best_label)
        print("     expanding %d nodes against Dijkstra's %d - a reduction of"
              % (best_expanded, dijkstra_expanded))
        print("     %.1f%% while returning exactly the same path cost. An"
              % (100.0 * (dijkstra_expanded - best_expanded)
                 / max(1, dijkstra_expanded)))
        print("     admissible heuristic changes HOW MUCH is searched, not the")
        print("     quality of the answer.")
    if fastest_label is not None and fastest_label != best_label:
        print("   * Fewest expansions overall: %s with %d nodes, but this is"
              % (fastest_label, fastest_expanded))
        print("     only a speed record - it carries no optimality guarantee.")
    if worst_cost_label is not None:
        print("   * %s returned a path costing %s instead of %s"
              % (worst_cost_label, format_cost(worst_cost),
                 format_cost(optimal_cost)))
        print("     (%.1f%% worse). This is the heuristic trade-off: by"
              % (100.0 * (worst_cost - optimal_cost) / optimal_cost))
        print("     trusting an over-weighted estimate the search finishes")
        print("     sooner but is no longer guaranteed to be optimal.")
    else:
        print("   * On this particular map even the weighted runs happened to")
        print("     return the optimal cost. Weighted A* is not guaranteed to")
        print("     do so - try built-in map 3, or switch on 8-way movement,")
        print("     to see it settle for a more expensive path.")




# #############################################################################
# #                                                                           #
# #  EMBEDDED TEST AND VALIDATION SUITE                                       #
# #                                                                           #
# #  Reachable from the main menu as option 9. Correctness is checked four    #
# #  ways: fixed hand-calculated cases, an INDEPENDENT reference search       #
# #  (reference_uniform_cost - linear scan, no heap, no heuristic, shares no  #
# #  code with A*), randomised stress testing against that reference, and     #
# #  property checks (legal path, costs add up, heap ordering, weighted-A*    #
# #  bound). Kept in this file so the whole of Problem 3 is one script.       #
# #                                                                           #
# #############################################################################

PASSED = 0
FAILED = 0
RESULTS = []      # (id, description, expected, actual, verdict)


# =============================================================================
# REFERENCE SEARCH - deliberately not A*
# =============================================================================
def reference_uniform_cost(grid, start, goal,
                           allow_diagonal=False,
                           allow_corner_cutting=False):
    # Uniform-cost search: linear scan for the minimum, no heap, no
    # heuristic. Slow on purpose, written to be obviously right rather than
    # fast. Returns the optimal cost, or None if the goal is unreachable.
    distance = {start: 0.0}
    settled = set()

    while True:
        current = None
        best = None
        for cell, dist in distance.items():
            if cell in settled:
                continue
            if best is None or dist < best:
                best = dist
                current = cell
        if current is None:
            return None              # nothing left to settle
        if current == goal:
            return best
        settled.add(current)

        r, c = current
        for nr, nc, step, _label in grid.neighbours(
                r, c, allow_diagonal, allow_corner_cutting):
            if (nr, nc) in settled:
                continue
            new_dist = best + step
            if (nr, nc) not in distance or new_dist < distance[(nr, nc)]:
                distance[(nr, nc)] = new_dist


# =============================================================================
# TEST HELPERS
# =============================================================================
def record(test_id, description, expected, actual, ok):
    global PASSED, FAILED
    if ok:
        PASSED += 1
        verdict = "PASS"
    else:
        FAILED += 1
        verdict = "FAIL"
    RESULTS.append((test_id, description, str(expected), str(actual), verdict))


def costs_equal(a, b):
    if a is None or b is None:
        return a is b or a == b
    return abs(a - b) < 1e-6


def check_path_validity(grid, result, start, goal,
                        allow_diagonal, allow_corner_cutting):
    # Structural checks that hold for any successful search.
    # Returns (ok, message).
    path = result['path']
    if not path:
        return False, "path is empty"
    if path[0] != start:
        return False, "path does not begin at the start cell"
    if path[-1] != goal:
        return False, "path does not end at the goal cell"

    total = 0.0
    for i in range(1, len(path)):
        pr, pc = path[i - 1]
        cr, cc = path[i]
        legal = False
        for nr, nc, step, _label in grid.neighbours(
                pr, pc, allow_diagonal, allow_corner_cutting):
            if (nr, nc) == (cr, cc):
                legal = True
                total += step
                break
        if not legal:
            return False, "illegal move from (%d,%d) to (%d,%d)" % (
                pr, pc, cr, cc)
    if not costs_equal(total, result['cost']):
        return False, ("reported cost %s does not match the sum of the "
                       "step costs %s" % (result['cost'], total))
    return True, "ok"


def build(rows):
    grid = Grid(rows, "test map")
    return grid, grid.start, grid.goal


# =============================================================================
# GROUP A : FIXED TEST CASES WITH HAND-CALCULATED ANSWERS
# =============================================================================
def test_fixed_cases():
    # --- TC1: straight line on an open 3x3 grid --------------------------
    grid, s, g = build([
        "S..",
        "...",
        "..E",
    ])
    res = astar(grid, s, g)
    record("TC1", "Open 3x3 grid, 4-way, shortest route", 4,
           format_cost(res['cost']), costs_equal(res['cost'], 4))

    # --- TC2: a wall forces a detour -------------------------------------
    grid, s, g = build([
        "S.#..",
        "..#..",
        "..#..",
        ".....",
        "....E",
    ])
    res = astar(grid, s, g)
    expected = reference_uniform_cost(grid, s, g)
    record("TC2", "Wall forces a detour, cost matches reference",
           format_cost(expected), format_cost(res['cost']),
           costs_equal(res['cost'], expected))

    # --- TC3: the longer route is the cheaper one -------------------------
    # Straight over the mountains costs 3 x 8; round the bottom is all plain.
    # A* has to take the detour, which shows g(n) is tracking cost and not
    # just counting steps.
    grid, s, g = build([
        "S^^^E",
        ".....",
        ".....",
    ])
    res = astar(grid, s, g)
    expected = reference_uniform_cost(grid, s, g)
    detoured = all(cell[0] != 0 or cell[1] in (0, 4) for cell in res['path'])
    record("TC3", "Costly terrain avoided in favour of a longer cheap route",
           "%s and detour taken" % format_cost(expected),
           "%s and detour %s" % (format_cost(res['cost']),
                                 "taken" if detoured else "NOT taken"),
           costs_equal(res['cost'], expected) and detoured)

    # --- TC4: unreachable goal -------------------------------------------
    grid, s, g = build([
        "S....",
        ".###.",
        ".#E#.",
        ".###.",
        ".....",
    ])
    res = astar(grid, s, g)
    record("TC4", "Sealed goal is correctly reported as unreachable",
           "found = False", "found = %s" % res['found'], res['found'] is False)

    # --- TC5: start equals goal ------------------------------------------
    grid, s, _ = build([
        "S..",
        "...",
    ])
    res = astar(grid, s, s)
    ok = res['found'] and costs_equal(res['cost'], 0) and len(res['path']) == 1
    record("TC5", "Start equals goal returns a zero cost single cell path",
           "cost 0, 1 cell",
           "cost %s, %d cell(s)" % (format_cost(res['cost']),
                                    len(res['path'])), ok)

    # --- TC6: diagonal movement uses sqrt(2) ------------------------------
    # corner to corner on a clear 3x3 is two diagonal moves
    grid, s, g = build([
        "S..",
        "...",
        "..E",
    ])
    res = astar(grid, s, g, heuristic_name='octile', allow_diagonal=True)
    record("TC6", "8-way movement charges sqrt(2) per diagonal step",
           "%.4f" % (2 * SQRT2), "%.4f" % res['cost'],
           costs_equal(res['cost'], 2 * SQRT2))

    # --- TC7: corner cutting rule ----------------------------------------
    # Both shoulders are walls, so the diagonal squeeze has to be refused.
    grid, s, g = build([
        "S#",
        "#E",
    ])
    blocked = astar(grid, s, g, heuristic_name='octile',
                    allow_diagonal=True, allow_corner_cutting=False)
    allowed = astar(grid, s, g, heuristic_name='octile',
                    allow_diagonal=True, allow_corner_cutting=True)
    ok = (blocked['found'] is False) and (allowed['found'] is True)
    record("TC7", "Corner cutting rule is enforced",
           "blocked = no path, allowed = path",
           "blocked = %s, allowed = %s"
           % ("no path" if not blocked['found'] else "path",
              "path" if allowed['found'] else "no path"), ok)

    # --- TC8: input validation rejects malformed maps ---------------------
    ragged = Grid.validate_rows(["S..", "...."])
    badchar = Grid.validate_rows(["S.X", "..E"])
    twostart = Grid.validate_rows(["SS.", "..E"])
    ok = (ragged is not None) and (badchar is not None) and (twostart is not None)
    record("TC8", "Malformed maps are rejected by the validator",
           "3 errors reported",
           "%d errors reported" % sum(1 for e in (ragged, badchar, twostart)
                                      if e is not None), ok)


# =============================================================================
# GROUP B : PROPERTY TESTS
# =============================================================================
def test_heap_property():
    # entries must come out in non-decreasing f order, whatever goes in
    random.seed(20260409)
    heap = MinHeap()
    for _ in range(4000):
        f = random.randint(0, 500) / 10.0
        heap.push(f, random.randint(0, 50), random.randint(0, 30),
                  random.randint(0, 30))
    previous = None
    ordered = True
    popped = 0
    while not heap.is_empty():
        entry = heap.pop()
        popped += 1
        if previous is not None and entry[0] < previous - EPS:
            ordered = False
            break
        previous = entry[0]
    record("TC9", "MinHeap releases 4000 random entries in sorted f order",
           "4000 entries, sorted",
           "%d entries, %s" % (popped, "sorted" if ordered else "OUT OF ORDER"),
           ordered and popped == 4000)


def test_heuristic_admissibility():
    # h must never exceed the real remaining cost. Checked directly: for
    # every passable cell, compare h against the exact cost to the goal from
    # the reference search. This is what the optimality guarantee rests on.
    random.seed(7)
    rows = [
        "..............",
        "..~~***~~.....",
        ".~~*^^^*~~....",
        ".~*^^^^^*~~...",
        ".~*^^^^^*~~...",
        ".~~*^^^*~~....",
        "..~~***~~.....",
        "..............",
    ]
    grid = Grid(rows, "admissibility map")
    goal = (7, 13)
    scale = grid.min_terrain_cost()

    violations = 0
    checked = 0
    for r in range(grid.height):
        for c in range(grid.width):
            if grid.is_wall(r, c):
                continue
            true_cost = reference_uniform_cost(grid, (r, c), goal)
            if true_cost is None:
                continue
            checked += 1
            h_value = heuristic('manhattan', r, c, goal[0], goal[1], scale)
            if h_value > true_cost + 1e-9:
                violations += 1
    record("TC10", "Manhattan h(n) never exceeds the true cost (4-way)",
           "0 violations in %d cells" % checked,
           "%d violations in %d cells" % (violations, checked),
           violations == 0)


def test_weighted_bound():
    # Weighted A* comes with a bound: cost <= w x optimal. The path can be
    # worse than optimal but never arbitrarily worse. Both halves checked.
    random.seed(1234)
    breaches = 0
    suboptimal_seen = 0
    trials = 0
    for _ in range(120):
        grid, start, goal = random_solvable_map(12, 16, 26)
        optimal = reference_uniform_cost(grid, start, goal)
        if optimal is None or optimal == 0:
            continue
        for weight in (1.5, 3.0, 6.0):
            res = astar(grid, start, goal, heuristic_name='manhattan',
                        weight=weight, record_expanded=False)
            trials += 1
            if not res['found']:
                breaches += 1
                continue
            if res['cost'] > weight * optimal + 1e-6:
                breaches += 1
            if res['cost'] > optimal + 1e-6:
                suboptimal_seen += 1
    record("TC11", "Weighted A* stays within the w x optimal bound",
           "0 breaches in %d runs" % trials,
           "%d breaches (%d runs returned a suboptimal path, as expected)"
           % (breaches, suboptimal_seen), breaches == 0)


# =============================================================================
# GROUP C : RANDOMISED STRESS TEST AGAINST THE REFERENCE SEARCH
# =============================================================================
def random_solvable_map(height, width, wall_pct):
    # keeps regenerating until the two corners are actually connected
    rough = ['~', '*', '^']
    while True:
        rows = []
        for _r in range(height):
            row = ""
            for _c in range(width):
                roll = random.randint(1, 100)
                if roll <= wall_pct:
                    row += '#'
                elif roll <= wall_pct + 25:
                    row += rough[random.randint(0, 2)]
                else:
                    row += '.'
            rows.append(row)
        grid = Grid(rows, "random")
        grid.cells[0][0] = '.'
        grid.cells[height - 1][width - 1] = '.'
        start = (0, 0)
        goal = (height - 1, width - 1)
        if goal in flood_fill_reachable(grid, start):
            return grid, start, goal


def test_random_stress(trials=250):
    random.seed(2026)
    mismatches = 0
    invalid = 0
    checked = 0
    detail = ""

    for _ in range(trials):
        height = random.randint(6, 16)
        width = random.randint(6, 18)
        wall_pct = random.randint(5, 32)
        grid, start, goal = random_solvable_map(height, width, wall_pct)

        for allow_diagonal in (False, True):
            h_name = 'octile' if allow_diagonal else 'manhattan'
            optimal = reference_uniform_cost(grid, start, goal,
                                             allow_diagonal, False)
            res = astar(grid, start, goal, heuristic_name=h_name,
                        allow_diagonal=allow_diagonal,
                        allow_corner_cutting=False,
                        record_expanded=False)
            checked += 1

            if optimal is None:
                if res['found']:
                    mismatches += 1
                continue
            if not res['found'] or not costs_equal(res['cost'], optimal):
                mismatches += 1
                if not detail:
                    detail = " (first mismatch on a %dx%d map)" % (height, width)
                continue
            ok, message = check_path_validity(grid, res, start, goal,
                                              allow_diagonal, False)
            if not ok:
                invalid += 1
                if not detail:
                    detail = " (%s)" % message

    record("TC12",
           "A* matches the independent reference search on random maps",
           "0 cost mismatches in %d searches" % checked,
           "%d mismatches%s" % (mismatches, detail), mismatches == 0)
    record("TC13",
           "Every returned path is structurally legal and self-consistent",
           "0 invalid paths",
           "%d invalid paths%s" % (invalid, detail), invalid == 0)


def test_heuristic_never_changes_the_answer():
    # Admissible heuristics all have to agree on the cost even though they
    # explore different amounts of the map. Separates "how fast" from
    # "how good", which is the whole point of the comparison feature.
    random.seed(99)
    disagreements = 0
    savings = []
    for _ in range(60):
        grid, start, goal = random_solvable_map(14, 18, 22)
        baseline = astar(grid, start, goal, heuristic_name='zero',
                         record_expanded=False)
        if not baseline['found']:
            continue
        for name in ('manhattan', 'euclidean', 'chebyshev'):
            res = astar(grid, start, goal, heuristic_name=name,
                        record_expanded=False)
            if not res['found'] or not costs_equal(res['cost'],
                                                   baseline['cost']):
                disagreements += 1
        best = astar(grid, start, goal, heuristic_name='manhattan',
                     record_expanded=False)
        if baseline['expanded'] > 0:
            savings.append(100.0 * (baseline['expanded'] - best['expanded'])
                           / baseline['expanded'])

    average_saving = sum(savings) / len(savings) if savings else 0.0
    record("TC14",
           "All admissible heuristics return the same optimal cost",
           "0 disagreements",
           "%d disagreements (Manhattan expanded %.1f%% fewer nodes than "
           "Dijkstra on average)" % (disagreements, average_saving),
           disagreements == 0)


# =============================================================================
# SELF-TEST RUNNER  (menu option 9)
# =============================================================================
def run_self_tests():
    """Run the built-in test and validation suite (menu option 9)."""
    global PASSED, FAILED, RESULTS
    PASSED = 0
    FAILED = 0
    RESULTS = []

    print("=" * 78)
    print(" CSC2103 PROBLEM 3 - A* SEARCH : AUTOMATED TEST AND VALIDATION")
    print("=" * 78)
    print(" Reference search : uniform-cost search with a linear minimum scan,")
    print("                    written independently of A* (no heuristic, no")
    print("                    heap) and used as the source of truth.")
    print("=" * 78)

    test_fixed_cases()
    test_heap_property()
    test_heuristic_admissibility()
    test_weighted_bound()
    test_random_stress()
    test_heuristic_never_changes_the_answer()

    print()
    print(" %-5s %-52s %s" % ("ID", "What is being verified", "Result"))
    print("-" * 78)
    for test_id, description, expected, actual, verdict in RESULTS:
        print(" %-5s %-52s %s" % (test_id, description[:52], verdict))
        print("       expected : %s" % expected)
        print("       actual   : %s" % actual)
    print("-" * 78)
    print(" TOTAL : %d passed, %d failed" % (PASSED, FAILED))
    print("=" * 78)

# =============================================================================
# MAIN MENU
# =============================================================================
def print_header():
    print(LINE)
    print(" CSC2103 DATA STRUCTURES AND ALGORITHMS - GROUP PROJECT")
    print(" PROBLEM 3 : HEURISTIC ALGORITHM")
    print(" A* (A-STAR) SEARCH ALGORITHM FOR PATH FINDING")
    print(LINE)
    print(" A* finds a route on a weighted grid by always expanding the node")
    print(" with the smallest  f(n) = g(n) + w * h(n)  value, where g is the")
    print(" cost already paid and h is the heuristic estimate of the cost")
    print(" still to be paid. The heuristic focuses the search towards the")
    print(" goal instead of spreading out evenly in every direction.")
    print(LINE)


def print_menu(grid, settings):
    print()
    print(THIN)
    if grid is None:
        print(" Current map : (none loaded)")
    else:
        print(" Current map : %s  [%d rows x %d columns]"
              % (grid.name, grid.height, grid.width))
    print(" Settings    : %s" % settings.describe())
    print(THIN)
    print("  1. Load a built-in sample map")
    print("  2. Load a map from a text file")
    print("  3. Enter a map manually")
    print("  4. Generate a random test map")
    print("  5. Show the current map and legend")
    print("  6. Configure search settings")
    print("  7. Run the A* search")
    print("  8. Compare heuristics (A* vs Dijkstra vs Weighted A*)")
    print("  9. Run the built-in test and validation suite")
    print(" 10. Exit")


def main():
    print_header()
    grid = None
    settings = Settings()

    while True:
        print_menu(grid, settings)
        choice = prompt(" Select an option: ").strip()

        if choice == '1':
            loaded = choose_builtin_map()
            if loaded is not None:
                grid = loaded
                render_grid(grid, grid.start, grid.goal, title="  MAP:")
        elif choice == '2':
            loaded = load_map_from_file()
            if loaded is not None:
                grid = loaded
                render_grid(grid, grid.start, grid.goal, title="  MAP:")
        elif choice == '3':
            loaded = enter_map_manually()
            if loaded is not None:
                grid = loaded
                render_grid(grid, grid.start, grid.goal, title="  MAP:")
        elif choice == '4':
            loaded = generate_random_map()
            if loaded is not None:
                grid = loaded
                render_grid(grid, grid.start, grid.goal, title="  MAP:")
        elif choice == '5':
            if grid is None:
                print("  ! No map is loaded yet. Use option 1 to 4 first.")
            else:
                render_grid(grid, grid.start, grid.goal,
                            title="  CURRENT MAP: %s" % grid.name)
                print_legend()
        elif choice == '6':
            configure_settings(settings)
        elif choice == '7':
            if grid is None:
                print("  ! No map is loaded yet. Use option 1 to 4 first.")
            else:
                run_search(grid, settings)
        elif choice == '8':
            if grid is None:
                print("  ! No map is loaded yet. Use option 1 to 4 first.")
            else:
                compare_algorithms(grid, settings)
        elif choice == '9':
            run_self_tests()
        elif choice == '10':
            print()
            print(" Program ended. Thank you.")
            break
        else:
            print("  ! Invalid option. Please choose a number from 1 to 10.")


if __name__ == '__main__':
    import sys
    # "python3 problem3_astar_pathfinding.py test" jumps straight to the
    # self-test suite; with no argument it opens the interactive menu.
    if len(sys.argv) > 1 and sys.argv[1].lower() in ('test', 'tests', '--test'):
        run_self_tests()
    else:
        main()
