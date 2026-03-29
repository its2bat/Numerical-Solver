"""Mesh generation for 2D and 3D FEM solvers using gmsh."""
import numpy as np
import gmsh
from .params import GeometryParams, MeshParams, MeshResult


_gmsh_initialized = False

def ensure_gmsh_initialized():
    """Initialize gmsh ONCE from the main thread. Must be called before
    any mesh building (which may run in a worker thread)."""
    global _gmsh_initialized
    if not _gmsh_initialized:
        try:
            gmsh.finalize()
        except Exception:
            pass
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        _gmsh_initialized = True


def _init_gmsh(model_name):
    """Prepare gmsh for a new model. Assumes gmsh is already initialized."""
    gmsh.clear()
    gmsh.model.add(model_name)


def _build_2d_geometry(geo, geom):
    """Build the 2D cross-section geometry. Returns tags dict."""
    xc = geom.xc
    R_in, R_tube, R_paste = geom.R_in, geom.R_tube, geom.R_paste
    W, H = geom.W, geom.H

    # Points
    pc = geo.addPoint(xc, 0.0, 0.0)
    p_in_top = geo.addPoint(xc, R_in, 0.0)
    p_in_right = geo.addPoint(xc + R_in, 0.0, 0.0)
    p_in_bot = geo.addPoint(xc, -R_in, 0.0)

    p_tube_top = geo.addPoint(xc, R_tube, 0.0)
    p_tube_right = geo.addPoint(xc + R_tube, 0.0, 0.0)
    p_tube_bot = geo.addPoint(xc, -R_tube, 0.0)

    p_paste_top = geo.addPoint(xc, R_paste, 0.0)
    p_paste_right = geo.addPoint(xc + R_paste, 0.0, 0.0)

    p_mold_br = geo.addPoint(W, 0.0, 0.0)
    p_mold_tr = geo.addPoint(W, H, 0.0)
    p_mold_tl = geo.addPoint(xc, H, 0.0)

    # Arcs
    arc_in_u = geo.addCircleArc(p_in_top, pc, p_in_right)
    arc_in_l = geo.addCircleArc(p_in_right, pc, p_in_bot)
    arc_tube_u = geo.addCircleArc(p_tube_top, pc, p_tube_right)
    arc_tube_l = geo.addCircleArc(p_tube_right, pc, p_tube_bot)
    arc_paste = geo.addCircleArc(p_paste_top, pc, p_paste_right)

    # Lines at y=0
    l_y0_tube = geo.addLine(p_in_right, p_tube_right)
    l_y0_paste = geo.addLine(p_tube_right, p_paste_right)
    l_y0_mold = geo.addLine(p_paste_right, p_mold_br)

    # Mold edges
    l_right = geo.addLine(p_mold_br, p_mold_tr)
    l_top = geo.addLine(p_mold_tr, p_mold_tl)

    # Symmetry lines
    l_sym_mold = geo.addLine(p_mold_tl, p_paste_top)
    l_sym_paste = geo.addLine(p_paste_top, p_tube_top)
    l_sym_tube_u = geo.addLine(p_tube_top, p_in_top)
    l_sym_tube_l = geo.addLine(p_tube_bot, p_in_bot)

    # Surfaces
    cl_tube_u = geo.addCurveLoop([arc_in_u, l_y0_tube, -arc_tube_u, l_sym_tube_u])
    s_tube_u = geo.addPlaneSurface([cl_tube_u])

    cl_tube_l = geo.addCurveLoop([arc_tube_l, l_sym_tube_l, -arc_in_l, l_y0_tube])
    s_tube_l = geo.addPlaneSurface([cl_tube_l])

    cl_paste = geo.addCurveLoop([arc_tube_u, l_y0_paste, -arc_paste, l_sym_paste])
    s_paste = geo.addPlaneSurface([cl_paste])

    cl_mold = geo.addCurveLoop([arc_paste, l_y0_mold, l_right, l_top, l_sym_mold])
    s_mold = geo.addPlaneSurface([cl_mold])

    return {
        "s_tube_u": s_tube_u, "s_tube_l": s_tube_l,
        "s_paste": s_paste, "s_mold": s_mold,
        "arcs": [arc_in_u, arc_in_l, arc_tube_u, arc_tube_l, arc_paste],
        "dirichlet_curves": [arc_in_u, arc_in_l],
        "robin_curves": [arc_tube_l, l_y0_paste, l_y0_mold, l_right, l_top],
        "sym_curves": [l_sym_mold, l_sym_paste, l_sym_tube_u, l_sym_tube_l],
    }


def _apply_mesh_fields(tags, mp):
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", tags["arcs"])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 200)

    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", mp.h_arc)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", mp.h_global)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", mp.dist_min)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", mp.dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)


def _extract_tris(surf_tag, tag_to_idx):
    etypes, _, enodes = gmsh.model.mesh.getElements(2, surf_tag)
    for etype, nodes in zip(etypes, enodes):
        if etype in (2, 9):
            nn = 3 if etype == 2 else 6
            conn = nodes.reshape(-1, nn)[:, :3].astype(int)
            return np.vectorize(tag_to_idx.get)(conn)
    return np.empty((0, 3), dtype=int)


def _edges_from_pg(phys_id, tag_to_idx):
    edges = []
    for ent in gmsh.model.getEntitiesForPhysicalGroup(1, phys_id):
        etypes, _, enodes = gmsh.model.mesh.getElements(1, ent)
        for etype, nodes in zip(etypes, enodes):
            if etype == 1:
                edges.append(np.vectorize(tag_to_idx.get)(
                    nodes.reshape(-1, 2).astype(int)))
            elif etype == 8:
                edges.append(np.vectorize(tag_to_idx.get)(
                    nodes.reshape(-1, 3)[:, :2].astype(int)))
    if not edges:
        return np.empty((0, 2), dtype=int)
    return np.unique(np.sort(np.vstack(edges), axis=1), axis=0)


def build_mesh_2d(geom: GeometryParams, mp: MeshParams,
                  log=None) -> MeshResult:
    """Generate 2D mesh with gmsh. Returns MeshResult."""
    if log:
        log("Building 2D mesh ...")

    _init_gmsh("mesh_2d_3mat")
    geo = gmsh.model.geo
    tags = _build_2d_geometry(geo, geom)
    geo.synchronize()

    # Physical groups
    pg_dir = gmsh.model.addPhysicalGroup(1, tags["dirichlet_curves"])
    pg_rob = gmsh.model.addPhysicalGroup(1, tags["robin_curves"])
    gmsh.model.addPhysicalGroup(1, tags["sym_curves"])

    gmsh.model.addPhysicalGroup(2, [tags["s_tube_u"]])
    gmsh.model.addPhysicalGroup(2, [tags["s_tube_l"]])
    gmsh.model.addPhysicalGroup(2, [tags["s_paste"]])
    gmsh.model.addPhysicalGroup(2, [tags["s_mold"]])

    _apply_mesh_fields(tags, mp)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Laplace2D")

    # Extract
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords_all = node_coords.reshape(-1, 3)[:, :2]
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    tris_tube_u = _extract_tris(tags["s_tube_u"], tag_to_idx)
    tris_tube_l = _extract_tris(tags["s_tube_l"], tag_to_idx)
    tris_paste = _extract_tris(tags["s_paste"], tag_to_idx)
    tris_mold = _extract_tris(tags["s_mold"], tag_to_idx)

    edges_dir = _edges_from_pg(pg_dir, tag_to_idx)
    edges_rob = _edges_from_pg(pg_rob, tag_to_idx)

    gmsh.finalize()

    # Active node reduction
    used = set()
    for arr in [tris_tube_u, tris_tube_l, tris_paste, tris_mold,
                edges_dir, edges_rob]:
        used |= set(arr.ravel())
    used = np.array(sorted(used), dtype=int)

    new_idx = -np.ones(coords_all.shape[0], dtype=int)
    new_idx[used] = np.arange(len(used))

    coords = coords_all[used]
    tris_tube_u = new_idx[tris_tube_u]
    tris_tube_l = new_idx[tris_tube_l]
    tris_paste = new_idx[tris_paste]
    tris_mold = new_idx[tris_mold]
    edges_dir = new_idx[edges_dir]
    edges_rob = new_idx[edges_rob]

    tris_tube = np.vstack([tris_tube_u, tris_tube_l])
    triangles = np.vstack([tris_tube, tris_paste, tris_mold])

    Nn = coords.shape[0]
    info = (f"Mesh: {Nn} nodes, {len(triangles)} triangles\n"
            f"  tube: {len(tris_tube)} ({len(tris_tube_u)} upper + {len(tris_tube_l)} lower)\n"
            f"  paste: {len(tris_paste)}, mold: {len(tris_mold)}\n"
            f"  Dirichlet edges: {len(edges_dir)}, Robin edges: {len(edges_rob)}")
    if log:
        log(info)

    return MeshResult(
        coords=coords, triangles=triangles,
        tris_tube=tris_tube, tris_paste=tris_paste, tris_mold=tris_mold,
        edges_dir=edges_dir, edges_rob=edges_rob,
        n_nodes=Nn, n_elements=len(triangles), info=info
    )


def build_mesh_3d(geom: GeometryParams, mp: MeshParams,
                  log=None) -> MeshResult:
    """Generate 3D mesh (extrude 2D cross-section along Z). Returns MeshResult."""
    if log:
        log("Building 3D mesh (extrusion) ...")

    _init_gmsh("mesh_3d_3mat")
    geo = gmsh.model.geo
    tags = _build_2d_geometry(geo, geom)
    geo.synchronize()

    # Extrude all 4 surfaces
    all_2d = [(2, tags["s_tube_u"]), (2, tags["s_tube_l"]),
              (2, tags["s_paste"]), (2, tags["s_mold"])]
    geo.extrude(all_2d, 0, 0, geom.Lz, [geom.nz_layers])
    geo.synchronize()

    _apply_mesh_fields(tags, mp)

    if log:
        log("Meshing 3D ...")
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.optimize("Laplace2D")

    # Extract nodes
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    P_all = node_coords.reshape(-1, 3)
    tag_to_idx = {int(t): i for i, t in enumerate(node_tags)}

    # All tets
    all_tets = []
    for dim, tag in gmsh.model.getEntities(3):
        etypes, _, enodes = gmsh.model.mesh.getElements(3, tag)
        for etype, nodes in zip(etypes, enodes):
            if etype in (4, 11):
                nn = 4 if etype == 4 else 10
                conn = nodes.reshape(-1, nn)[:, :4].astype(int)
                all_tets.append(np.vectorize(tag_to_idx.get)(conn))
    all_tets = np.vstack(all_tets)

    # All boundary triangles
    all_tris = []
    for dim, tag in gmsh.model.getEntities(2):
        etypes, _, enodes = gmsh.model.mesh.getElements(2, tag)
        for etype, nodes in zip(etypes, enodes):
            if etype in (2, 9):
                nn = 3 if etype == 2 else 6
                conn = nodes.reshape(-1, nn)[:, :3].astype(int)
                all_tris.append(np.vectorize(tag_to_idx.get)(conn))
    all_tris = np.vstack(all_tris) if all_tris else np.empty((0, 3), dtype=int)

    gmsh.finalize()

    # Active node reduction
    used = set(all_tets.ravel()) | set(all_tris.ravel())
    used = np.array(sorted(used), dtype=int)
    new_idx = -np.ones(P_all.shape[0], dtype=int)
    new_idx[used] = np.arange(len(used))

    P = P_all[used]
    all_tets = new_idx[all_tets]
    all_tris_reindexed = new_idx[all_tris]
    Nn = P.shape[0]

    # Classify tets by material (centroid radius)
    xc = geom.xc
    R_in, R_tube, R_paste = geom.R_in, geom.R_tube, geom.R_paste
    Lz = geom.Lz

    tet_centroids = P[all_tets].mean(axis=1)
    r_tet = np.sqrt((tet_centroids[:, 0] - xc)**2 + tet_centroids[:, 1]**2)

    r_mid_tube_paste = 0.5 * (R_tube + R_paste)
    mask_tube = r_tet < r_mid_tube_paste
    mask_paste = (~mask_tube) & (r_tet < R_paste + 0.3 * (R_paste - R_tube))
    mask_mold = ~mask_tube & ~mask_paste

    tets_tube = all_tets[mask_tube]
    tets_paste = all_tets[mask_paste]
    tets_mold = all_tets[mask_mold]

    # Classify boundary triangles
    tri_c = P[all_tris_reindexed].mean(axis=1)
    r_tri = np.sqrt((tri_c[:, 0] - xc)**2 + tri_c[:, 1]**2)

    tol_r = 5e-6
    tol_c = 5e-10

    is_dirichlet = np.isclose(r_tri, R_in, atol=tol_r)
    is_sym = np.isclose(tri_c[:, 0], xc, atol=tol_c) & ~is_dirichlet
    is_z0 = np.isclose(tri_c[:, 2], 0.0, atol=tol_c)
    is_zL = np.isclose(tri_c[:, 2], Lz, atol=tol_c)
    is_z_end = (is_z0 | is_zL) & ~is_dirichlet & ~is_sym

    not_special = ~is_dirichlet & ~is_sym & ~is_z_end
    W = geom.W
    H = geom.H
    is_mold_right = not_special & np.isclose(tri_c[:, 0], W, atol=tol_c)
    is_mold_top = not_special & np.isclose(tri_c[:, 1], H, atol=tol_c)
    is_mold_bot = not_special & np.isclose(tri_c[:, 1], 0.0, atol=tol_c) & (r_tri > R_paste - tol_r)
    is_paste_bot = (not_special & np.isclose(tri_c[:, 1], 0.0, atol=tol_c) &
                    (r_tri > R_tube - tol_r) & (r_tri < R_paste + tol_r) & ~is_mold_bot)
    is_tube_exposed = not_special & np.isclose(r_tri, R_tube, atol=tol_r) & (tri_c[:, 1] < -tol_c)

    is_robin = is_z_end | is_mold_right | is_mold_top | is_mold_bot | is_paste_bot | is_tube_exposed

    tri_dir = all_tris_reindexed[is_dirichlet]
    tri_rob = all_tris_reindexed[is_robin]

    info = (f"Mesh: {Nn} nodes, {len(all_tets)} tets\n"
            f"  tube: {len(tets_tube)}, paste: {len(tets_paste)}, mold: {len(tets_mold)}\n"
            f"  Dirichlet: {len(tri_dir)}, Robin: {len(tri_rob)}, "
            f"Sym: {int(is_sym.sum())}")
    if log:
        log(info)

    return MeshResult(
        coords=P, all_tets=all_tets,
        tets_tube=tets_tube, tets_paste=tets_paste, tets_mold=tets_mold,
        tri_dir=tri_dir, tri_rob=tri_rob,
        n_nodes=Nn, n_elements=len(all_tets), info=info
    )
