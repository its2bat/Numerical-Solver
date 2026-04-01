"""Parameter dataclasses for FEM heat transfer solvers."""
from dataclasses import dataclass, field


@dataclass
class GeometryParams:
    W_mm: float = 1.4
    H_mm: float = 1.0
    tube_OD_mm: float = 0.8
    tube_ID_mm: float = 0.5
    paste_thickness_mm: float = 0.1
    Lz_mm: float = 1.0       # 3D only
    nz_layers: int = 28       # 3D only

    @property
    def R_in_mm(self):
        return self.tube_ID_mm / 2.0

    @property
    def R_tube_mm(self):
        return self.tube_OD_mm / 2.0

    @property
    def R_paste_mm(self):
        return self.R_tube_mm + self.paste_thickness_mm

    # SI conversions
    @property
    def W(self): return self.W_mm * 1e-3
    @property
    def H(self): return self.H_mm * 1e-3
    @property
    def Lz(self): return self.Lz_mm * 1e-3
    @property
    def R_in(self): return self.R_in_mm * 1e-3
    @property
    def R_tube(self): return self.R_tube_mm * 1e-3
    @property
    def R_paste(self): return self.R_paste_mm * 1e-3
    @property
    def xc(self): return self.W / 2.0

    def validate(self):
        errs = []
        if self.tube_ID_mm >= self.tube_OD_mm:
            errs.append("Tube ID must be < OD")
        if self.R_paste_mm * 1e-3 > self.W / 2 + 1e-15:
            errs.append(f"R_paste ({self.R_paste_mm}mm) > W/2 ({self.W_mm/2}mm)")
        if self.R_paste_mm * 1e-3 > self.H * 1e3 * 1e-3 + 1e-15:
            errs.append(f"R_paste ({self.R_paste_mm}mm) > H ({self.H_mm}mm)")
        if self.paste_thickness_mm <= 0:
            errs.append("Paste thickness must be > 0")
        return errs


@dataclass
class MaterialParams:
    k_tube: float = 16.0
    rho_tube: float = 8000.0
    cp_tube: float = 500.0
    k_paste: float = 9.0
    rho_paste: float = 2500.0
    cp_paste: float = 800.0
    k_solder: float = 50.0
    rho_solder: float = 8500.0
    cp_solder: float = 180.0


@dataclass
class BCParams:
    T_wall: float = 60.0
    h_conv: float = 15.0
    Tinf: float = 25.0
    T0: float = 25.0       # transient IC
    bc_inner: str = "dirichlet"   # "dirichlet" or "neumann"
    q_flux: float = 1000.0        # W/m² heat flux (Neumann only)


@dataclass
class MeshParams:
    h_global: float = 0.06e-3
    h_arc: float = 0.008e-3
    h_mid: float = 0.015e-3
    dist_min: float = 0.02e-3
    dist_max: float = 0.15e-3


@dataclass
class SolverParams:
    dt: float = 0.05
    t_end: float = 500.0
    conv_tol: float = 1e-3


@dataclass
class MeshResult:
    """Holds extracted mesh data."""
    coords: object = None        # 2D: (Nn,2), 3D: (Nn,3)
    triangles: object = None     # 2D: all tris combined
    tris_tube: object = None
    tris_paste: object = None
    tris_mold: object = None
    edges_dir: object = None     # 2D boundary edges
    edges_rob: object = None
    # 3D specific
    all_tets: object = None
    tets_tube: object = None
    tets_paste: object = None
    tets_mold: object = None
    tri_dir: object = None       # 3D boundary triangles
    tri_rob: object = None
    # Stats
    n_nodes: int = 0
    n_elements: int = 0
    info: str = ""


@dataclass
class SolveResult:
    """Holds solver output."""
    T: object = None                 # solution on half-domain
    T_full: object = None            # mirrored full-domain
    coords_full: object = None       # mirrored coords
    tris_full: object = None         # 2D: mirrored triangles
    tets_full: object = None         # 3D: mirrored tets
    T_min: float = 0.0
    T_max: float = 0.0
    q_in: float = 0.0
    q_out: float = 0.0
    # Transient extras
    tmax_hist: object = None
    times_arr: object = None
    converged: bool = False
    conv_step: int = -1
    T_history: list = field(default_factory=list)
    times_anim: list = field(default_factory=list)
    dT_hist: object = None   # residual history for convergence plot
    info: str = ""
