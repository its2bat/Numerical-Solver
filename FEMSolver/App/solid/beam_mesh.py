"""
Structured triangular mesh generation for beam analysis.

Coordinate system
-----------------
  x : along beam length (0 → L)
  y : transverse (0 → W)
  z : out-of-plane thickness t (not meshed — plane stress assumption)

Cell sizing strategy
--------------------
All mesh functions use ``ny`` (cells across the beam WIDTH) as the primary
resolution parameter and derive the x-direction count automatically so that
cells are approximately square (dx ≈ dy = W/ny).  This prevents the severe
shear-locking that P1 triangles exhibit on elongated cells in bending.

Straight cantilever
-------------------
  Fixed edge : x = 0
  Tip edge   : x = L

Zig-zag (serpentine) cantilever
--------------------------------
  n_segs horizontal beam segments stacked vertically.
  Segments are connected at alternating right / left ends via short
  connector rectangles of width W and height g (the gap).

  Fixed edge : left edge (x = 0) of segment 0
  Tip  edge  : free end of last segment
               → x = L  if n_segs is odd
               → x = 0  if n_segs is even

  The connector cell size matches the beam cell size (dx_beam = W/ny),
  ensuring a conforming (compatible) mesh at segment–connector interfaces.
"""

import numpy as np
from scipy.spatial import cKDTree


# ── helpers ──────────────────────────────────────────────────────────────────

def _mesh_rect(x0, y0, w, h, nx, ny):
    """Structured triangular mesh for rectangle [x0,x0+w]×[y0,y0+h].

    Each of the nx×ny quads is split into 2 triangles via the
    bottom-left → top-right diagonal (CCW orientation).

    Returns
    -------
    coords : (Nn,2) — (ny+1)×(nx+1) node array in row-major order
    tris   : (2·nx·ny, 3) — triangle connectivity
    """
    xs = np.linspace(x0, x0 + w, nx + 1)
    ys = np.linspace(y0, y0 + h, ny + 1)
    XX, YY = np.meshgrid(xs, ys)
    coords = np.column_stack([XX.ravel(), YY.ravel()])

    def idx(i, j):          # col i (x), row j (y) → node index
        return j * (nx + 1) + i

    tris = []
    for j in range(ny):
        for i in range(nx):
            a, b = idx(i, j),   idx(i+1, j)
            c, d = idx(i+1, j+1), idx(i, j+1)
            tris.append([a, b, c])   # lower-right: CCW ✓
            tris.append([a, c, d])   # upper-left:  CCW ✓

    return coords, np.array(tris, dtype=int)


def _merge_nodes(coords, tris, tol=1e-10):
    """Merge coincident nodes (shared edges of adjacent rectangles)."""
    Nn = len(coords)
    parent = np.arange(Nn)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    tree = cKDTree(coords)
    for i, j in tree.query_pairs(tol):
        pi, pj = find(i), find(j)
        if pi != pj:
            parent[pi] = pj

    # Build compact node list
    root_to_new = {}
    new_coords = []
    mapping = np.empty(Nn, dtype=int)
    for i in range(Nn):
        r = find(i)
        if r not in root_to_new:
            root_to_new[r] = len(new_coords)
            new_coords.append(coords[r])
        mapping[i] = root_to_new[r]

    new_coords = np.array(new_coords)
    new_tris = mapping[tris]

    # Drop degenerate triangles
    mask = ((new_tris[:, 0] != new_tris[:, 1]) &
            (new_tris[:, 1] != new_tris[:, 2]) &
            (new_tris[:, 0] != new_tris[:, 2]))
    return new_coords, new_tris[mask]


# ── public API ───────────────────────────────────────────────────────────────

def mesh_straight_cantilever(L, W, ny=10, log=None):
    """Structured triangular mesh for a straight cantilever beam.

    Cell size is set to dx = dy = W/ny so that elements are approximately
    square, which prevents shear locking in P1 triangle bending problems.

    Parameters
    ----------
    L, W  : beam length and width (SI, metres)
    ny    : cells across the beam width (default 10)
            nx (along length) is computed automatically: nx = round(L/W * ny)

    Returns
    -------
    coords      : (Nn, 2) node coordinates
    tris        : (Ne, 3) triangle connectivity
    fixed_nodes : node indices at x = 0 (clamped edge)
    tip_nodes   : node indices at x = L (free tip)
    """
    dx = W / ny
    nx = max(2, round(L / dx))

    if log:
        log(f"  Straight cantilever mesh: nx={nx}, ny={ny} "
            f"(dx={dx*1e6:.2f} µm, dy={dx*1e6:.2f} µm)")
        log(f"  Elements: {2*nx*ny},  Nodes: {(nx+1)*(ny+1)}")

    coords, tris = _mesh_rect(0.0, 0.0, L, W, nx, ny)

    tol = dx * 1e-6
    fixed_nodes = np.where(np.abs(coords[:, 0]) < tol)[0]
    tip_nodes   = np.where(np.abs(coords[:, 0] - L) < tol)[0]

    if log:
        log(f"  Fixed nodes: {len(fixed_nodes)},  Tip nodes: {len(tip_nodes)}")

    return coords, tris, fixed_nodes, tip_nodes


def mesh_zigzag_cantilever(L, W, g, n_segs, ny=10, log=None):
    """Structured triangular mesh for a zig-zag (serpentine) cantilever.

    The cell size dx = W/ny is used uniformly for all segments and connectors,
    ensuring a conformal (compatible) mesh at every interface.

    Parameters
    ----------
    L       : length of each horizontal segment
    W       : beam width (segments and connectors share this value)
    g       : gap between adjacent segments (connector height)
    n_segs  : number of horizontal segments (≥ 1)
    ny      : cells across the beam width (controls overall resolution)

    Geometry layout
    ---------------
    Seg 0 fixed at x = 0.  Connectors alternate right / left:
      right connector (even index i): x ∈ [L-W, L]
      left  connector (odd  index i): x ∈ [0,   W]

    Tip end of last segment:
      n_segs odd  → x = L  (right edge)
      n_segs even → x = 0  (left edge)

    Returns
    -------
    coords, tris, fixed_nodes, tip_nodes
    """
    if n_segs < 1:
        raise ValueError("n_segs must be ≥ 1")

    dx      = W / ny                      # uniform cell size
    nx      = max(2, round(L / dx))       # cells along segment length
    nx_conn = ny                          # cells across connector (W/dx = ny)
    ny_conn = max(1, round(g / dx))       # cells across gap

    if log:
        log(f"  Zig-zag mesh: {n_segs} segments")
        log(f"  dx={dx*1e6:.2f} µm  nx_seg={nx}  nx_conn={nx_conn}  ny_conn={ny_conn}")

    all_coords = []
    all_tris   = []
    offset     = 0

    # ---- segments ----------------------------------------------------------
    for i in range(n_segs):
        y0 = i * (W + g)
        c, t = _mesh_rect(0.0, y0, L, W, nx, ny)
        all_coords.append(c)
        all_tris.append(t + offset)
        offset += len(c)

    # ---- connectors --------------------------------------------------------
    for i in range(n_segs - 1):
        y_bot = i * (W + g) + W          # top of segment i
        x0 = (L - W) if (i % 2 == 0) else 0.0
        c, t = _mesh_rect(x0, y_bot, W, g, nx_conn, ny_conn)
        all_coords.append(c)
        all_tris.append(t + offset)
        offset += len(c)

    # ---- merge coincident nodes (at shared edges) -------------------------
    all_coords = np.vstack(all_coords)
    all_tris   = np.vstack(all_tris)
    tol_merge  = dx * 1e-6
    coords, tris = _merge_nodes(all_coords, all_tris, tol=tol_merge)

    # ---- boundary sets ----------------------------------------------------
    tol = dx * 1e-6

    # Fixed: left edge (x=0) of segment 0, y in [0, W]
    fixed_nodes = np.where(
        (np.abs(coords[:, 0]) < tol) &
        (coords[:, 1] >= -tol) &
        (coords[:, 1] <=  W + tol)
    )[0]

    # Tip: free end of last segment
    last_y0 = (n_segs - 1) * (W + g)
    tip_x   = L if (n_segs % 2 == 1) else 0.0
    tip_nodes = np.where(
        (np.abs(coords[:, 0] - tip_x) < tol) &
        (coords[:, 1] >= last_y0 - tol) &
        (coords[:, 1] <= last_y0 + W + tol)
    )[0]

    if log:
        total_h = n_segs * W + (n_segs - 1) * g
        log(f"  Total height ≈ {total_h*1e6:.1f} µm")
        log(f"  Nodes: {len(coords)},  Elements: {len(tris)}")
        log(f"  Fixed nodes: {len(fixed_nodes)},  Tip nodes: {len(tip_nodes)}")

    return coords, tris, fixed_nodes, tip_nodes
