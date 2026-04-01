"""
FEM Heat Transfer Solver
Desktop application with wizard-style step-by-step interface.
"""
import sys
import os
import numpy as np
import traceback

# Set matplotlib backend before any other import
import matplotlib
matplotlib.use("QtAgg")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit,
    QProgressBar, QMessageBox, QFileDialog, QSplitter,
    QSizePolicy, QFrame, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QFont, QIcon, QPalette, QColor

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.tri as mtri

# ---- Solver imports ----
from solvers.params import (GeometryParams, MaterialParams, BCParams,
                            MeshParams, SolverParams, MeshResult, SolveResult)
from solvers.mesh_builder import build_mesh_2d, build_mesh_3d
from solvers.steady_2d import solve_steady_2d
from solvers.transient_2d import solve_transient_2d
from solvers.steady_3d import solve_steady_3d
from solvers.transient_3d import solve_transient_3d


# =====================================================================
# REUSABLE WIDGETS
# =====================================================================

class LogConsole(QPlainTextEdit):
    """Read-only log console with monospace font."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setMaximumBlockCount(2000)

    def log(self, text):
        self.appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        QApplication.processEvents()


class MplCanvas(QWidget):
    """Matplotlib figure embedded in Qt with navigation toolbar and hover readout."""
    def __init__(self, parent=None, width=8, height=6):
        super().__init__(parent)
        self.fig = Figure(figsize=(width, height), dpi=100,
                          facecolor="#2a2a2a")
        self.ax = self.fig.add_subplot(111)
        self._style_ax()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Hover status bar
        self._hover_label = QLabel("  Move cursor over plot  ")
        self._hover_label.setFont(QFont("Consolas", 9))
        self._hover_label.setStyleSheet(
            "background:#1a1a1a; color:#00e5ff; "
            "padding:3px 8px; border-top:1px solid #444;")
        self._hover_label.setMinimumHeight(22)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self._hover_label)

        # Interpolator state (set externally for temperature lookup)
        self._interp = None
        self._canvas_cid = self.canvas.mpl_connect(
            "motion_notify_event", self._on_hover)

    def _style_ax(self):
        self.ax.set_facecolor("#1e1e1e")
        self.ax.tick_params(colors="#ccc")
        self.ax.xaxis.label.set_color("#ccc")
        self.ax.yaxis.label.set_color("#ccc")
        self.ax.title.set_color("#ddd")
        for spine in self.ax.spines.values():
            spine.set_color("#555")

    def clear(self):
        self._interp = None
        self.ax.clear()
        self._style_ax()
        self._hover_label.setText("  Move cursor over plot  ")
        self.canvas.draw()

    def set_interpolator(self, triang, T_values):
        """Attach a LinearTriInterpolator for temperature hover readout."""
        from matplotlib.tri import LinearTriInterpolator
        self._interp = LinearTriInterpolator(triang, T_values)

    def _on_hover(self, event):
        if event.inaxes != self.ax or self._interp is None:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                T_raw = self._interp(x, y)
            # LinearTriInterpolator returns a masked array; check mask
            import numpy.ma as ma
            if ma.is_masked(T_raw) or T_raw is ma.masked:
                self._hover_label.setText(
                    f"  x={x*1e3:.4f} mm   y={y*1e3:.4f} mm   "
                    f"T = -- (outside domain)")
            else:
                T_val = float(T_raw)
                self._hover_label.setText(
                    f"  x={x*1e3:.4f} mm   y={y*1e3:.4f} mm   "
                    f"T = {T_val:.4f} °C")
        except Exception:
            pass


class StepIndicator(QWidget):
    """Step progress indicator bar."""
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.labels = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        for i, name in enumerate(steps):
            lbl = QLabel(f"  {i+1}. {name}  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                              "background-color: #444; color: #999;")
            layout.addWidget(lbl)
            self.labels.append(lbl)
            if i < len(steps) - 1:
                arrow = QLabel("->")
                arrow.setAlignment(Qt.AlignmentFlag.AlignCenter)
                arrow.setStyleSheet("color: #666; font-weight: bold;")
                layout.addWidget(arrow)

    def set_active(self, idx):
        for i, lbl in enumerate(self.labels):
            if i < idx:
                lbl.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                                  "background-color: #2E7D32; color: white; "
                                  "font-weight: bold;")
            elif i == idx:
                lbl.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                                  "background-color: #1565C0; color: white; "
                                  "font-weight: bold;")
            else:
                lbl.setStyleSheet("padding: 4px 8px; border-radius: 4px; "
                                  "background-color: #444; color: #999;")


# =====================================================================
# BACKGROUND WORKERS
# =====================================================================

class MeshWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, is_3d, geom, mp):
        super().__init__()
        self.is_3d = is_3d
        self.geom = geom
        self.mp = mp

    def run(self):
        try:
            if self.is_3d:
                result = build_mesh_3d(self.geom, self.mp,
                                       log=self.log_signal.emit)
            else:
                result = build_mesh_2d(self.geom, self.mp,
                                       log=self.log_signal.emit)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Mesh generation failed:\n{traceback.format_exc()}")


class SolveWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float)  # step, total, dT, Tmax
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, solver_type, mesh, mat, bc, geom, sp=None):
        super().__init__()
        self.solver_type = solver_type
        self.mesh = mesh
        self.mat = mat
        self.bc = bc
        self.geom = geom
        self.sp = sp

    def run(self):
        try:
            log = self.log_signal.emit

            def prog(step, total, dT, Tmax):
                self.progress_signal.emit(step, total, dT, Tmax)

            if self.solver_type == "2d_steady":
                result = solve_steady_2d(self.mesh, self.mat, self.bc,
                                         self.geom, log=log)
            elif self.solver_type == "2d_transient":
                result = solve_transient_2d(self.mesh, self.mat, self.bc,
                                            self.geom, self.sp,
                                            log=log, progress=prog)
            elif self.solver_type == "3d_steady":
                result = solve_steady_3d(self.mesh, self.mat, self.bc,
                                         self.geom, log=log)
            elif self.solver_type == "3d_transient":
                result = solve_transient_3d(self.mesh, self.mat, self.bc,
                                            self.geom, self.sp,
                                            log=log, progress=prog)
            else:
                raise ValueError(f"Unknown solver type: {self.solver_type}")

            self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"Solver failed:\n{traceback.format_exc()}")


# =====================================================================
# PAGE 1: WELCOME / SOLVER SELECTION
# =====================================================================

class WelcomePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Title
        title = QLabel("FEM Heat Transfer Solver")
        title.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #64B5F6; margin: 20px;")
        layout.addWidget(title)

        subtitle = QLabel("3-Material Microtube Heat Exchanger — FEM Solver")
        subtitle.setFont(QFont("Segoe UI", 11))
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #aaa;")
        layout.addWidget(subtitle)

        layout.addSpacing(30)

        # Dimension selection
        dim_group = QGroupBox("Dimension")
        dim_group.setFont(QFont("Segoe UI", 11))
        dim_layout = QHBoxLayout(dim_group)
        self.btn_2d = QRadioButton("2D Cross-Section")
        self.btn_3d = QRadioButton("3D Extruded")
        self.btn_2d.setChecked(True)
        self.dim_group = QButtonGroup()
        self.dim_group.addButton(self.btn_2d, 0)
        self.dim_group.addButton(self.btn_3d, 1)
        dim_layout.addWidget(self.btn_2d)
        dim_layout.addWidget(self.btn_3d)
        layout.addWidget(dim_group)

        # Type selection
        type_group = QGroupBox("Analysis Type")
        type_group.setFont(QFont("Segoe UI", 11))
        type_layout = QHBoxLayout(type_group)
        self.btn_steady = QRadioButton("Steady-State")
        self.btn_transient = QRadioButton("Transient")
        self.btn_steady.setChecked(True)
        self.type_group = QButtonGroup()
        self.type_group.addButton(self.btn_steady, 0)
        self.type_group.addButton(self.btn_transient, 1)
        type_layout.addWidget(self.btn_steady)
        type_layout.addWidget(self.btn_transient)
        layout.addWidget(type_group)

        # Description
        self.desc = QLabel()
        self.desc.setFont(QFont("Segoe UI", 10))
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color: #ccc; padding: 10px; "
                                "background: #333; border-radius: 5px;")
        layout.addWidget(self.desc)
        self._update_desc()

        self.dim_group.buttonClicked.connect(lambda: self._update_desc())
        self.type_group.buttonClicked.connect(lambda: self._update_desc())

        layout.addStretch()

    def _update_desc(self):
        dim = "2D" if self.btn_2d.isChecked() else "3D"
        typ = "Steady" if self.btn_steady.isChecked() else "Transient"
        descs = {
            "2D Steady": "Solves the 2D heat equation on the cross-section. "
                         "Direct sparse solve (spsolve). Fastest option.",
            "2D Transient": "Backward Euler time-stepping on 2D cross-section. "
                            "Auto-stops when convergence tolerance is met. "
                            "Generates animation GIF.",
            "3D Steady": "Extrudes 2D cross-section along tube axis. "
                         "~150k nodes, ~840k tets. Direct sparse solve.",
            "3D Transient": "Full 3D transient with Backward Euler. "
                            "Pre-factorized LU for fast time steps. "
                            "Largest computation (~1 min factorize).",
        }
        self.desc.setText(f"{dim} {typ}:\n{descs[f'{dim} {typ}']}")

    @property
    def is_3d(self):
        return self.btn_3d.isChecked()

    @property
    def is_transient(self):
        return self.btn_transient.isChecked()

    @property
    def solver_type(self):
        dim = "3d" if self.is_3d else "2d"
        typ = "transient" if self.is_transient else "steady"
        return f"{dim}_{typ}"


# =====================================================================
# PAGE 2: PARAMETER INPUT
# =====================================================================

MATERIAL_PRESETS = {
    "— Select Preset —": None,
    "Copper + Thermal Paste + Solder": dict(
        k_tube=385.0, rho_tube=8960.0, cp_tube=385.0,
        k_paste=6.0, rho_paste=2500.0, cp_paste=800.0,
        k_solder=50.0, rho_solder=8500.0, cp_solder=180.0),
    "Steel + Thermal Paste + FR4": dict(
        k_tube=16.0, rho_tube=8000.0, cp_tube=500.0,
        k_paste=3.0, rho_paste=2200.0, cp_paste=900.0,
        k_solder=0.3, rho_solder=1850.0, cp_solder=1100.0),
    "Aluminum + Thermal Grease + Epoxy": dict(
        k_tube=205.0, rho_tube=2700.0, cp_tube=897.0,
        k_paste=4.0, rho_paste=2400.0, cp_paste=850.0,
        k_solder=1.1, rho_solder=1200.0, cp_solder=1000.0),
    "Titanium + Phase Change Material + Aluminum": dict(
        k_tube=22.0, rho_tube=4500.0, cp_tube=520.0,
        k_paste=0.7, rho_paste=900.0, cp_paste=2000.0,
        k_solder=160.0, rho_solder=2700.0, cp_solder=900.0),
}


class ParametersPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._is_3d = False
        self._is_transient = False

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        title = QLabel("Set Parameters")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Scrollable grid of parameter groups
        grid = QGridLayout()
        grid.setSpacing(10)

        # ---- Geometry ----
        geo_box = QGroupBox("Geometry (mm)")
        geo_form = QFormLayout(geo_box)
        self.sp_W = self._spin(1.4, 0.1, 100.0, 2)
        self.sp_H = self._spin(1.0, 0.1, 100.0, 2)
        self.sp_OD = self._spin(0.8, 0.05, 50.0, 3)
        self.sp_ID = self._spin(0.5, 0.05, 50.0, 3)
        self.sp_paste = self._spin(0.1, 0.01, 10.0, 3)
        self.sp_Lz = self._spin(1.0, 0.1, 100.0, 2)
        self.sp_nz = QSpinBox(); self.sp_nz.setRange(4, 200); self.sp_nz.setValue(28)
        self.lbl_Lz = QLabel("Tube length Lz:")
        self.lbl_nz = QLabel("Z layers:")

        geo_form.addRow("Domain width W:", self.sp_W)
        geo_form.addRow("Domain height H:", self.sp_H)
        geo_form.addRow("Tube OD:", self.sp_OD)
        geo_form.addRow("Tube ID:", self.sp_ID)
        geo_form.addRow("Paste thickness:", self.sp_paste)
        geo_form.addRow(self.lbl_Lz, self.sp_Lz)
        geo_form.addRow(self.lbl_nz, self.sp_nz)
        grid.addWidget(geo_box, 0, 0)

        # ---- Materials ----
        mat_box = QGroupBox("Materials")
        mat_form = QFormLayout(mat_box)

        self.preset_combo = QComboBox()
        for name in MATERIAL_PRESETS:
            self.preset_combo.addItem(name)
        self.preset_combo.currentTextChanged.connect(self._apply_preset)
        mat_form.addRow("Material Preset:", self.preset_combo)

        self.sp_k_tube = self._spin(16.0, 0.1, 500.0, 1)
        self.sp_rho_tube = self._spin(8000.0, 100.0, 30000.0, 0)
        self.sp_cp_tube = self._spin(500.0, 50.0, 5000.0, 0)
        self.sp_k_paste = self._spin(9.0, 0.1, 500.0, 1)
        self.sp_rho_paste = self._spin(2500.0, 100.0, 30000.0, 0)
        self.sp_cp_paste = self._spin(800.0, 50.0, 5000.0, 0)
        self.sp_k_solder = self._spin(50.0, 0.1, 500.0, 1)
        self.sp_rho_solder = self._spin(8500.0, 100.0, 30000.0, 0)
        self.sp_cp_solder = self._spin(180.0, 10.0, 5000.0, 0)

        mat_form.addRow("k_tube (W/mK):", self.sp_k_tube)
        self.lbl_rho_t = QLabel("rho_tube (kg/m3):")
        mat_form.addRow(self.lbl_rho_t, self.sp_rho_tube)
        self.lbl_cp_t = QLabel("cp_tube (J/kgK):")
        mat_form.addRow(self.lbl_cp_t, self.sp_cp_tube)

        mat_form.addRow("k_paste (W/mK):", self.sp_k_paste)
        self.lbl_rho_p = QLabel("rho_paste:")
        mat_form.addRow(self.lbl_rho_p, self.sp_rho_paste)
        self.lbl_cp_p = QLabel("cp_paste:")
        mat_form.addRow(self.lbl_cp_p, self.sp_cp_paste)

        mat_form.addRow("k_solder (W/mK):", self.sp_k_solder)
        self.lbl_rho_s = QLabel("rho_solder:")
        mat_form.addRow(self.lbl_rho_s, self.sp_rho_solder)
        self.lbl_cp_s = QLabel("cp_solder:")
        mat_form.addRow(self.lbl_cp_s, self.sp_cp_solder)

        # Store rho/cp labels+spins for hiding
        self._transient_widgets = [
            self.lbl_rho_t, self.sp_rho_tube, self.lbl_cp_t, self.sp_cp_tube,
            self.lbl_rho_p, self.sp_rho_paste, self.lbl_cp_p, self.sp_cp_paste,
            self.lbl_rho_s, self.sp_rho_solder, self.lbl_cp_s, self.sp_cp_solder,
        ]
        grid.addWidget(mat_box, 0, 1)

        # ---- Boundary Conditions ----
        bc_box = QGroupBox("Boundary Conditions")
        bc_form = QFormLayout(bc_box)
        self.sp_Twall = self._spin(60.0, -100.0, 1000.0, 1)
        self.sp_h = self._spin(15.0, 0.1, 10000.0, 1)
        self.sp_Tinf = self._spin(25.0, -100.0, 1000.0, 1)
        self.sp_T0 = self._spin(25.0, -100.0, 1000.0, 1)
        self.lbl_T0 = QLabel("Initial T0 (C):")

        # BC type for inner cylinder
        bc_type_layout = QHBoxLayout()
        self.btn_dirichlet = QRadioButton("T_wall (Dirichlet)")
        self.btn_neumann = QRadioButton("Heat Flux q_in (Neumann)")
        self.btn_dirichlet.setChecked(True)
        self.bc_type_group = QButtonGroup()
        self.bc_type_group.addButton(self.btn_dirichlet, 0)
        self.bc_type_group.addButton(self.btn_neumann, 1)
        bc_type_layout.addWidget(self.btn_dirichlet)
        bc_type_layout.addWidget(self.btn_neumann)

        self.sp_qflux = QDoubleSpinBox()
        self.sp_qflux.setRange(0.1, 1e7)
        self.sp_qflux.setDecimals(1)
        self.sp_qflux.setValue(1000.0)
        self.sp_qflux.setVisible(False)
        self.lbl_qflux = QLabel("q_in (W/m²):")
        self.lbl_qflux.setVisible(False)

        self.btn_neumann.toggled.connect(lambda checked: (
            self.sp_Twall.setEnabled(not checked),
            self.sp_qflux.setVisible(checked),
            self.lbl_qflux.setVisible(checked)
        ))

        bc_form.addRow("Inner BC:", bc_type_layout)
        bc_form.addRow("T_wall (°C):", self.sp_Twall)
        bc_form.addRow(self.lbl_qflux, self.sp_qflux)
        bc_form.addRow("h_conv (W/m2K):", self.sp_h)
        bc_form.addRow("T_inf (C):", self.sp_Tinf)
        bc_form.addRow(self.lbl_T0, self.sp_T0)
        grid.addWidget(bc_box, 1, 0)

        # ---- Solver settings ----
        solver_box = QGroupBox("Solver Settings")
        solver_form = QFormLayout(solver_box)
        self.sp_dt = self._spin(0.01, 1e-5, 100.0, 5)
        self.sp_tend = self._spin(500.0, 1.0, 100000.0, 1)
        self.sp_tol = self._spin(0.001, 1e-8, 1.0, 6)
        self.lbl_dt = QLabel("dt (s):")
        self.lbl_tend = QLabel("t_end (s):")
        self.lbl_tol = QLabel("conv_tol (C):")

        solver_form.addRow(self.lbl_dt, self.sp_dt)
        solver_form.addRow(self.lbl_tend, self.sp_tend)
        solver_form.addRow(self.lbl_tol, self.sp_tol)

        self._solver_widgets = [
            self.lbl_dt, self.sp_dt,
            self.lbl_tend, self.sp_tend,
            self.lbl_tol, self.sp_tol,
        ]
        grid.addWidget(solver_box, 1, 1)

        layout.addLayout(grid)
        layout.addStretch()

    def _spin(self, val, lo, hi, dec):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setValue(val)
        sp.setMinimumWidth(100)
        return sp

    def update_visibility(self, is_3d, is_transient):
        self._is_3d = is_3d
        self._is_transient = is_transient

        # 3D-only fields
        for w in [self.sp_Lz, self.sp_nz, self.lbl_Lz, self.lbl_nz]:
            w.setVisible(is_3d)

        # Transient-only fields
        for w in self._transient_widgets + self._solver_widgets + [self.lbl_T0, self.sp_T0]:
            w.setVisible(is_transient)

    def get_params(self):
        geom = GeometryParams(
            W_mm=self.sp_W.value(), H_mm=self.sp_H.value(),
            tube_OD_mm=self.sp_OD.value(), tube_ID_mm=self.sp_ID.value(),
            paste_thickness_mm=self.sp_paste.value(),
            Lz_mm=self.sp_Lz.value(), nz_layers=self.sp_nz.value()
        )
        mat = MaterialParams(
            k_tube=self.sp_k_tube.value(), rho_tube=self.sp_rho_tube.value(),
            cp_tube=self.sp_cp_tube.value(),
            k_paste=self.sp_k_paste.value(), rho_paste=self.sp_rho_paste.value(),
            cp_paste=self.sp_cp_paste.value(),
            k_solder=self.sp_k_solder.value(), rho_solder=self.sp_rho_solder.value(),
            cp_solder=self.sp_cp_solder.value(),
        )
        bc = BCParams(
            T_wall=self.sp_Twall.value(), h_conv=self.sp_h.value(),
            Tinf=self.sp_Tinf.value(), T0=self.sp_T0.value(),
            bc_inner="neumann" if self.btn_neumann.isChecked() else "dirichlet",
            q_flux=self.sp_qflux.value()
        )
        mp = MeshParams()  # use defaults for mesh sizing
        sp = SolverParams(
            dt=self.sp_dt.value(), t_end=self.sp_tend.value(),
            conv_tol=self.sp_tol.value()
        )
        return geom, mat, bc, mp, sp

    def validate(self):
        geom, _, _, _, _ = self.get_params()
        return geom.validate()

    def _apply_preset(self, name):
        p = MATERIAL_PRESETS.get(name)
        if p is None:
            return
        self.sp_k_tube.setValue(p["k_tube"])
        self.sp_rho_tube.setValue(p["rho_tube"])
        self.sp_cp_tube.setValue(p["cp_tube"])
        self.sp_k_paste.setValue(p["k_paste"])
        self.sp_rho_paste.setValue(p["rho_paste"])
        self.sp_cp_paste.setValue(p["cp_paste"])
        self.sp_k_solder.setValue(p["k_solder"])
        self.sp_rho_solder.setValue(p["rho_solder"])
        self.sp_cp_solder.setValue(p["cp_solder"])


# =====================================================================
# PAGE 3: MESH GENERATION
# =====================================================================

class MeshPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Mesh Generation")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_generate = QPushButton("Generate Mesh")
        self.btn_generate.setFont(QFont("Segoe UI", 11))
        self.btn_generate.setMinimumHeight(35)
        self.btn_generate.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; "
            "border-radius: 5px; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #ccc; }")
        ctrl.addWidget(self.btn_generate)
        self.btn_3d_mesh = QPushButton("View 3D Mesh (PyVista)")
        self.btn_3d_mesh.setMinimumHeight(35)
        self.btn_3d_mesh.setStyleSheet(
            "QPushButton { background-color: #7B1FA2; color: white; "
            "border-radius: 5px; padding: 8px 16px; }"
            "QPushButton:hover { background-color: #6A1B9A; }"
            "QPushButton:disabled { background-color: #555; }")
        self.btn_3d_mesh.setVisible(False)
        ctrl.addWidget(self.btn_3d_mesh)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)
        layout.addLayout(ctrl)

        # Split: plot (big) + log (small)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot = MplCanvas(width=8, height=6)
        splitter.addWidget(self.plot)
        self.console = LogConsole()
        self.console.setFixedHeight(100)
        splitter.addWidget(self.console)
        splitter.setSizes([999, 100])
        layout.addWidget(splitter)

        # Mesh stats
        self.stats_label = QLabel("")
        self.stats_label.setFont(QFont("Consolas", 9))
        self.stats_label.setStyleSheet("padding: 4px 8px; background: #1e1e1e; "
                                       "color: #0f0; border-radius: 3px;")
        layout.addWidget(self.stats_label)

    def show_mesh_2d(self, mesh, geom):
        ax = self.plot.ax
        ax.clear()
        coords = mesh.coords
        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], mesh.triangles)
        ax.triplot(triang, linewidth=0.2, color="#5a8a5a")

        # Material interfaces
        xc = geom.xc
        theta_u = np.linspace(0, np.pi / 2, 200)
        theta_l = np.linspace(-np.pi / 2, 0, 200)
        for r_val, clr in [(geom.R_tube, "red"), (geom.R_paste, "blue")]:
            ax.plot(xc + r_val * np.cos(theta_u), r_val * np.sin(theta_u),
                    color=clr, linewidth=1.5, alpha=0.8)
        ax.plot(xc + geom.R_tube * np.cos(theta_l),
                geom.R_tube * np.sin(theta_l),
                color="red", linewidth=1.5, alpha=0.8)

        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"2D Mesh: {mesh.n_nodes:,} nodes, {mesh.n_elements:,} triangles  "
                     f"| Red=tube/air  Blue=paste/mold")
        ax.grid(True, alpha=0.2)
        self.plot.fig.tight_layout()
        self.plot.canvas.draw()

        # For mesh page, show x/y position only (override hover to show coords)
        self.plot._interp = None  # no T data yet
        # Patch hover to show just coordinates
        def _mesh_hover(event):
            if event.inaxes != self.plot.ax:
                return
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                xc = geom.xc
                r = np.sqrt((x - xc)**2 + y**2) * 1e3
                self.plot._hover_label.setText(
                    f"  x={x*1e3:.4f} mm   y={y*1e3:.4f} mm   "
                    f"r={r:.4f} mm from tube centre")
        self.plot.canvas.mpl_disconnect(self.plot._canvas_cid)
        self.plot._canvas_cid = self.plot.canvas.mpl_connect(
            "motion_notify_event", _mesh_hover)

    def show_mesh_3d_info(self, mesh):
        """For 3D, just show stats (interactive 3D mesh would need pyvista)."""
        ax = self.plot.ax
        ax.clear()
        ax.text(0.5, 0.5,
                f"3D Mesh Generated\n\n"
                f"Nodes: {mesh.n_nodes:,}\n"
                f"Tetrahedra: {mesh.n_elements:,}\n\n"
                f"Tube: {len(mesh.tets_tube):,} tets\n"
                f"Paste: {len(mesh.tets_paste):,} tets\n"
                f"Mold: {len(mesh.tets_mold):,} tets\n\n"
                f"Dirichlet faces: {len(mesh.tri_dir):,}\n"
                f"Robin faces: {len(mesh.tri_rob):,}",
                transform=ax.transAxes, fontsize=13,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        ax.set_axis_off()
        ax.set_title("3D Mesh Summary")
        self.plot.canvas.draw()


# =====================================================================
# PAGE 4: SOLVE
# =====================================================================

class SolvePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Solve")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        # Controls
        ctrl = QHBoxLayout()
        self.btn_solve = QPushButton("Start Solver")
        self.btn_solve.setFont(QFont("Segoe UI", 11))
        self.btn_solve.setMinimumHeight(35)
        self.btn_solve.setStyleSheet(
            "QPushButton { background-color: #FF9800; color: white; "
            "border-radius: 5px; padding: 8px 20px; }"
            "QPushButton:hover { background-color: #F57C00; }"
            "QPushButton:disabled { background-color: #ccc; }")
        ctrl.addWidget(self.btn_solve)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)
        self.step_label = QLabel("")
        self.step_label.setFont(QFont("Consolas", 9))
        ctrl.addWidget(self.step_label)
        layout.addLayout(ctrl)

        # Split: plot (big) + log (small)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot = MplCanvas(width=8, height=6)
        splitter.addWidget(self.plot)
        self.console = LogConsole()
        self.console.setFixedHeight(100)
        splitter.addWidget(self.console)
        splitter.setSizes([999, 100])
        layout.addWidget(splitter)

        # Store for hover interpolation
        self._triang_full = None
        self._T_full = None

    def show_result_2d(self, result, geom, bc):
        ax = self.plot.ax
        ax.clear()
        triang = mtri.Triangulation(result.coords_full[:, 0],
                                     result.coords_full[:, 1],
                                     result.tris_full)
        cf = ax.tricontourf(triang, result.T_full, levels=40, cmap="turbo")
        self.plot.fig.colorbar(cf, ax=ax, pad=0.03, label="Temperature (C)")
        ax.triplot(triang, linewidth=0.06, alpha=0.10, color="k")

        xc = geom.xc
        R_tube, R_paste = geom.R_tube, geom.R_paste
        theta_top = np.linspace(0, np.pi, 400)
        theta_bot = np.linspace(-np.pi, 0, 400)
        for r_val in [R_tube, R_paste]:
            ax.plot(xc + r_val * np.cos(theta_top), r_val * np.sin(theta_top),
                    "w--", linewidth=0.8, alpha=0.7)
        ax.plot(xc + R_tube * np.cos(theta_bot), R_tube * np.sin(theta_bot),
                "w--", linewidth=0.8, alpha=0.7)

        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(f"Temperature  min={result.T_min:.4f}  max={result.T_max:.4f} °C  "
                     f"(T_wall={bc.T_wall}°C)")
        ax.grid(True, alpha=0.2)
        pad = 0.00015
        ax.set_xlim(-pad, geom.W + pad)
        ax.set_ylim(-geom.R_tube - pad, geom.H + pad)
        self.plot.fig.tight_layout()
        self.plot.canvas.draw()

        # Attach hover interpolator
        self.plot.set_interpolator(triang, result.T_full)

    def init_live_convergence(self, bc, sp):
        """Two-panel ANSYS-style convergence monitor."""
        self.plot.fig.clear()
        self._ax_res, self._ax_tmax = self.plot.fig.subplots(2, 1, sharex=True)

        # Top panel: residual (log scale)
        ax = self._ax_res
        ax.set_facecolor("#1e1e1e")
        ax.set_ylabel("max|ΔT| (°C)", color="#ccc")
        ax.set_title("Convergence Monitor", color="#ddd")
        ax.axhline(sp.conv_tol, color="#ff5252", ls="--", alpha=0.8,
                   label=f"tol = {sp.conv_tol:.0e}")
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8, facecolor="#333333", labelcolor="#ccc")
        ax.grid(True, alpha=0.3, which="both")
        ax.tick_params(colors="#ccc")
        for spine in ax.spines.values():
            spine.set_color("#555555")

        # Bottom panel: Tmax
        ax2 = self._ax_tmax
        ax2.set_facecolor("#1e1e1e")
        ax2.set_xlabel("Time Step", color="#ccc")
        ax2.set_ylabel("Tmax (°C)", color="#ccc")
        ax2.axhline(bc.T_wall, color="#ef5350", ls="--", alpha=0.7,
                    label=f"T_wall={bc.T_wall}°C")
        ax2.axhline(bc.Tinf, color="#42a5f5", ls="--", alpha=0.7,
                    label=f"T_inf={bc.Tinf}°C")
        ax2.legend(loc="upper right", fontsize=8, facecolor="#333333", labelcolor="#ccc")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(colors="#ccc")
        for spine in ax2.spines.values():
            spine.set_color("#555555")

        self._res_line, = self._ax_res.plot([], [], color="#ff9800", lw=1.2)
        self._tmax_line, = self._ax_tmax.plot([], [], color="#69f0ae", lw=1.2)
        self._live_steps = []
        self._live_dT = []
        self._live_T = []
        self.plot.fig.patch.set_facecolor("#2a2a2a")
        self.plot.fig.tight_layout()
        self.plot.canvas.draw()

    def update_live_convergence(self, step, dT, Tmax):
        """Update both panels with new data point."""
        if not hasattr(self, '_res_line'):
            return
        self._live_steps.append(step)
        if dT > 0 and np.isfinite(dT):
            self._live_dT.append(dT)
        else:
            self._live_dT.append(self._live_dT[-1] if self._live_dT else 1e6)
        self._live_T.append(Tmax)

        self._res_line.set_xdata(self._live_steps)
        self._res_line.set_ydata(self._live_dT)
        self._tmax_line.set_xdata(self._live_steps)
        self._tmax_line.set_ydata(self._live_T)

        self._ax_res.relim(); self._ax_res.autoscale_view()
        self._ax_tmax.relim(); self._ax_tmax.autoscale_view()
        self.plot.canvas.draw_idle()
        QApplication.processEvents()

    def show_convergence_plot(self, result, bc, sp=None):
        self.plot.fig.clear()
        ax_res, ax_tmax = self.plot.fig.subplots(2, 1, sharex=True)

        steps = np.arange(len(result.tmax_hist))

        # Top: residual
        if hasattr(result, 'dT_hist') and result.dT_hist is not None:
            res = result.dT_hist
        else:
            res = np.abs(np.diff(result.tmax_hist, prepend=result.tmax_hist[0]))
            res[0] = res[1] if len(res) > 1 else 1.0

        for ax in [ax_res, ax_tmax]:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#ccc")
            for spine in ax.spines.values():
                spine.set_color("#555555")
            ax.grid(True, alpha=0.3, which="both")

        steps_r = np.arange(len(res))
        ax_res.semilogy(steps_r, np.maximum(res, 1e-12), color="#ff9800", lw=1.2)
        if sp is not None:
            ax_res.axhline(sp.conv_tol, color="#ff5252", ls="--", alpha=0.8,
                           label=f"tol={sp.conv_tol:.0e}")
        ax_res.set_ylabel("max|ΔT| (°C)", color="#ccc")
        ax_res.set_title("Convergence Monitor", color="#ddd")
        ax_res.legend(fontsize=8, facecolor="#333333", labelcolor="#ccc")

        ax_tmax.plot(steps, result.tmax_hist, color="#69f0ae", lw=1.2)
        ax_tmax.axhline(bc.T_wall, color="#ef5350", ls="--", alpha=0.7,
                        label=f"T_wall={bc.T_wall}°C")
        ax_tmax.axhline(bc.Tinf, color="#42a5f5", ls="--", alpha=0.7,
                        label=f"T_inf={bc.Tinf}°C")
        if result.converged:
            ax_tmax.axvline(result.conv_step, color="#69f0ae", ls="-.", alpha=0.7,
                            label=f"converged step {result.conv_step}")
        ax_tmax.set_xlabel("Time Step", color="#ccc")
        ax_tmax.set_ylabel("Tmax (°C)", color="#ccc")
        ax_tmax.legend(fontsize=8, facecolor="#333333", labelcolor="#ccc")

        self.plot.fig.patch.set_facecolor("#2a2a2a")
        self.plot.fig.tight_layout()
        self.plot.canvas.draw()

    def show_result_3d_info(self, result):
        ax = self.plot.ax
        ax.clear()
        info = (f"3D Solution Complete\n\n"
                f"T_min = {result.T_min:.4f} C\n"
                f"T_max = {result.T_max:.4f} C\n")
        if result.converged:
            info += f"\nConverged at t = {result.times_arr[-1]:.3f} s"
        if result.q_in != 0:
            info += f"\nHeat in: {result.q_in:.6e} W"
        info += ("\n\nUse 'View 3D' button on the Results page\n"
                 "for interactive PyVista visualization.")
        ax.text(0.5, 0.5, info,
                transform=ax.transAxes, fontsize=13,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
        ax.set_axis_off()
        self.plot.canvas.draw()


# =====================================================================
# PAGE 5: RESULTS
# =====================================================================

class ResultsPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        title = QLabel("Results Summary")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        layout.addWidget(title)

        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFont("Consolas", 10))
        self.summary.setMaximumHeight(200)
        layout.addWidget(self.summary)

        # Action buttons
        btn_row = QHBoxLayout()

        self.btn_save = QPushButton("Save Plots")
        self.btn_save.setMinimumHeight(35)
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2196F3; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        btn_row.addWidget(self.btn_save)

        self.btn_3d_view = QPushButton("View 3D (PyVista)")
        self.btn_3d_view.setMinimumHeight(35)
        self.btn_3d_view.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        btn_row.addWidget(self.btn_3d_view)

        self.btn_gif = QPushButton("Generate Animation GIF")
        self.btn_gif.setMinimumHeight(35)
        self.btn_gif.setStyleSheet(
            "QPushButton { background-color: #E91E63; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        btn_row.addWidget(self.btn_gif)

        self.btn_restart = QPushButton("Start Over")
        self.btn_restart.setMinimumHeight(35)
        self.btn_restart.setStyleSheet(
            "QPushButton { background-color: #607D8B; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        btn_row.addWidget(self.btn_restart)

        self.btn_pin = QPushButton("Pin This Run")
        self.btn_pin.setMinimumHeight(35)
        self.btn_pin.setStyleSheet(
            "QPushButton { background-color: #00897B; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        btn_row.addWidget(self.btn_pin)

        self.btn_compare = QPushButton("Compare Pinned")
        self.btn_compare.setMinimumHeight(35)
        self.btn_compare.setStyleSheet(
            "QPushButton { background-color: #F57F17; color: white; "
            "border-radius: 5px; padding: 8px 16px; }")
        self.btn_compare.setEnabled(False)
        btn_row.addWidget(self.btn_compare)

        self._pinned_runs = []  # list of (label, SolveResult, BCParams, MaterialParams)

        layout.addLayout(btn_row)

        # Plot area for final result
        self.plot = MplCanvas(width=7, height=5)
        layout.addWidget(self.plot)

        # Console for save/export feedback
        self.console = LogConsole()
        self.console.setMaximumHeight(80)
        layout.addWidget(self.console)


# =====================================================================
# MAIN WINDOW
# =====================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FEM Heat Transfer Solver")
        self.setMinimumSize(900, 700)
        self.resize(1050, 780)

        # State
        self.mesh_result = None
        self.solve_result = None
        self._worker = None

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(5)

        # Step indicator
        self.steps = StepIndicator(
            ["Select", "Parameters", "Mesh", "Solve", "Results"])
        main_layout.addWidget(self.steps)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #555;")
        main_layout.addWidget(line)

        # Stacked pages
        self.stack = QStackedWidget()
        self.page_welcome = WelcomePage()
        self.page_params = ParametersPage()
        self.page_mesh = MeshPage()
        self.page_solve = SolvePage()
        self.page_results = ResultsPage()

        self.stack.addWidget(self.page_welcome)
        self.stack.addWidget(self.page_params)
        self.stack.addWidget(self.page_mesh)
        self.stack.addWidget(self.page_solve)
        self.stack.addWidget(self.page_results)
        main_layout.addWidget(self.stack)

        # Navigation buttons
        nav = QHBoxLayout()
        self.btn_back = QPushButton("< Back")
        self.btn_back.setMinimumHeight(35)
        self.btn_back.setFont(QFont("Segoe UI", 10))
        nav.addWidget(self.btn_back)
        nav.addStretch()
        self.btn_next = QPushButton("Next >")
        self.btn_next.setMinimumHeight(35)
        self.btn_next.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.btn_next.setStyleSheet(
            "QPushButton { background-color: #1565C0; color: white; "
            "border-radius: 5px; padding: 8px 30px; }"
            "QPushButton:hover { background-color: #0D47A1; }"
            "QPushButton:disabled { background-color: #ccc; }")
        nav.addWidget(self.btn_next)
        main_layout.addLayout(nav)

        # Connections
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next.clicked.connect(self._go_next)
        self.page_mesh.btn_generate.clicked.connect(self._generate_mesh)
        self.page_solve.btn_solve.clicked.connect(self._run_solver)
        self.page_results.btn_restart.clicked.connect(self._restart)
        self.page_results.btn_save.clicked.connect(self._save_plots)
        self.page_results.btn_3d_view.clicked.connect(self._view_3d)
        self.page_results.btn_gif.clicked.connect(self._generate_gif)
        self.page_results.btn_pin.clicked.connect(self._pin_run)
        self.page_results.btn_compare.clicked.connect(self._compare_runs)

        self._update_page(0)

    def _current_page(self):
        return self.stack.currentIndex()

    def _update_page(self, idx):
        self.stack.setCurrentIndex(idx)
        self.steps.set_active(idx)
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setEnabled(idx < 4)
        if idx == 4:
            self.btn_next.setVisible(False)
        else:
            self.btn_next.setVisible(True)

        # Update visibility based on solver type
        if idx == 1:
            self.page_params.update_visibility(
                self.page_welcome.is_3d, self.page_welcome.is_transient)

        # Enable/disable next based on completion
        if idx == 2 and self.mesh_result is None:
            self.btn_next.setEnabled(False)
        if idx == 3 and self.solve_result is None:
            self.btn_next.setEnabled(False)

        # Update results page button visibility
        if idx == 4:
            is_3d = self.page_welcome.is_3d
            is_transient = self.page_welcome.is_transient
            self.page_results.btn_3d_view.setVisible(is_3d)
            self.page_results.btn_gif.setVisible(is_transient)

    def _go_back(self):
        idx = self._current_page()
        if idx > 0:
            if idx == 3:
                self.solve_result = None
            self._update_page(idx - 1)

    def _go_next(self):
        idx = self._current_page()
        if idx == 1:
            errs = self.page_params.validate()
            if errs:
                QMessageBox.warning(self, "Validation Error", "\n".join(errs))
                return
            self.mesh_result = None
            self.solve_result = None
        if idx < 4:
            self._update_page(idx + 1)

    def _generate_mesh(self):
        geom, mat, bc, mp, sp = self.page_params.get_params()
        is_3d = self.page_welcome.is_3d

        self.page_mesh.btn_generate.setEnabled(False)
        self.page_mesh.progress.setVisible(True)
        self.page_mesh.console.clear()
        self.mesh_result = None
        self.btn_next.setEnabled(False)

        # Re-initialize gmsh in main thread before every mesh run
        # (signal handlers require main thread; gmsh.clear() is unreliable in threads)
        from solvers.mesh_builder import ensure_gmsh_initialized
        ensure_gmsh_initialized()

        self._worker = MeshWorker(is_3d, geom, mp)
        self._worker.log_signal.connect(self.page_mesh.console.log)
        self._worker.finished.connect(self._on_mesh_done)
        self._worker.error.connect(self._on_mesh_error)
        self._worker.start()

    def _on_mesh_done(self, mesh_result):
        self.mesh_result = mesh_result
        self.page_mesh.progress.setVisible(False)
        self.page_mesh.btn_generate.setEnabled(True)
        self.page_mesh.stats_label.setText(mesh_result.info)
        self.btn_next.setEnabled(True)

        geom, _, _, _, _ = self.page_params.get_params()
        if self.page_welcome.is_3d:
            self.page_mesh.show_mesh_3d_info(mesh_result)
            self.page_mesh.btn_3d_mesh.setVisible(True)
            try:
                self.page_mesh.btn_3d_mesh.clicked.disconnect()
            except Exception:
                pass
            self.page_mesh.btn_3d_mesh.clicked.connect(self._view_3d_mesh)
        else:
            self.page_mesh.btn_3d_mesh.setVisible(False)
            self.page_mesh.show_mesh_2d(mesh_result, geom)

        self.page_mesh.console.log("Mesh generation complete!")

    def _on_mesh_error(self, err_msg):
        self.page_mesh.progress.setVisible(False)
        self.page_mesh.btn_generate.setEnabled(True)
        self.page_mesh.console.log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Mesh Error", err_msg)

    def _run_solver(self):
        if self.mesh_result is None:
            QMessageBox.warning(self, "No Mesh", "Generate mesh first!")
            return

        geom, mat, bc, mp, sp = self.page_params.get_params()
        solver_type = self.page_welcome.solver_type
        is_transient = self.page_welcome.is_transient
        self._sp = sp   # store for live convergence time axis

        self.page_solve.btn_solve.setEnabled(False)
        self.page_solve.progress.setVisible(True)
        if is_transient:
            n_max = int(round(sp.t_end / sp.dt))
            self.page_solve.progress.setRange(0, n_max)
            self.page_solve.init_live_convergence(bc, sp)
        else:
            self.page_solve.progress.setRange(0, 0)  # indeterminate
        self.page_solve.console.clear()
        self.solve_result = None
        self.btn_next.setEnabled(False)

        self._worker = SolveWorker(solver_type, self.mesh_result,
                                   mat, bc, geom, sp)
        self._worker.log_signal.connect(self.page_solve.console.log)
        self._worker.progress_signal.connect(self._on_solve_progress)
        self._worker.finished.connect(self._on_solve_done)
        self._worker.error.connect(self._on_solve_error)
        self._worker.start()

    def _on_solve_progress(self, step, total, dT, Tmax):
        self.page_solve.progress.setValue(step)
        self.page_solve.step_label.setText(
            f"Step {step}/{total}  Tmax={Tmax:.4f}C")
        if hasattr(self, '_sp') and hasattr(self.page_solve, '_res_line'):
            self.page_solve.update_live_convergence(step, dT, Tmax)

    def _on_solve_done(self, solve_result):
        self.solve_result = solve_result
        self.page_solve.progress.setVisible(False)
        self.page_solve.btn_solve.setEnabled(True)
        self.btn_next.setEnabled(True)

        geom, _, bc, _, sp = self.page_params.get_params()
        is_3d = self.page_welcome.is_3d
        is_transient = self.page_welcome.is_transient

        if is_3d:
            if is_transient:
                self.page_solve.show_convergence_plot(solve_result, bc, sp)
                # clear live state so old lines don't linger
                if hasattr(self.page_solve, '_res_line'):
                    del self.page_solve._res_line
            else:
                self.page_solve.show_result_3d_info(solve_result)
        elif is_transient:
            # Show convergence graph on solve page for transient
            self.page_solve.show_convergence_plot(solve_result, bc, sp)
        else:
            self.page_solve.show_result_2d(solve_result, geom, bc)

        self.page_solve.console.log("Solver complete!")
        self.page_solve.console.log(solve_result.info)

        # Pre-fill results page
        self._populate_results()

    def _on_solve_error(self, err_msg):
        self.page_solve.progress.setVisible(False)
        self.page_solve.btn_solve.setEnabled(True)
        self.page_solve.console.log(f"ERROR: {err_msg}")
        QMessageBox.critical(self, "Solver Error", err_msg)

    def _populate_results(self):
        r = self.solve_result
        geom, mat, bc, _, sp = self.page_params.get_params()
        solver_type = self.page_welcome.solver_type
        is_transient = self.page_welcome.is_transient

        summary = f"Solver: {solver_type.replace('_', ' ').upper()}\n"
        summary += f"Mesh: {self.mesh_result.n_nodes:,} nodes, "
        summary += f"{self.mesh_result.n_elements:,} elements\n"
        summary += f"Materials: tube k={mat.k_tube}, paste k={mat.k_paste}, "
        summary += f"solder k={mat.k_solder} W/mK\n"
        summary += f"BC: T_wall={bc.T_wall}C, h={bc.h_conv} W/m2K, "
        summary += f"T_inf={bc.Tinf}C\n\n"
        summary += f"T_min = {r.T_min:.4f} C\n"
        summary += f"T_max = {r.T_max:.4f} C\n"
        if r.q_in != 0:
            summary += f"Heat in: {r.q_in:.6e} W\n"
        if is_transient:
            if r.converged:
                summary += f"\nConverged at step {r.conv_step}, "
                summary += f"t = {r.times_arr[-1]:.3f} s"
            else:
                summary += f"\nDid not converge (t_end={sp.t_end}s)"

        self.page_results.summary.setPlainText(summary)

        # Show final temperature contour on results page (always show T field)
        if not self.page_welcome.is_3d:
            ax = self.page_results.plot.ax
            ax.clear()
            self.page_results.plot._style_ax()

            triang = mtri.Triangulation(r.coords_full[:, 0],
                                        r.coords_full[:, 1],
                                        r.tris_full)
            cf = ax.tricontourf(triang, r.T_full, levels=40, cmap="turbo")
            cbar = self.page_results.plot.fig.colorbar(cf, ax=ax, pad=0.03,
                                                       label="T (°C)")
            cbar.ax.yaxis.label.set_color("#ccc")
            cbar.ax.tick_params(colors="#ccc")

            # Tube/paste interface lines
            xc = geom.xc
            R_tube, R_paste = geom.R_tube, geom.R_paste
            theta_top = np.linspace(0, np.pi, 400)
            theta_bot = np.linspace(-np.pi, 0, 400)
            for rv in [R_tube, R_paste]:
                ax.plot(xc + rv * np.cos(theta_top), rv * np.sin(theta_top),
                        "w--", lw=0.8, alpha=0.6)
            ax.plot(xc + R_tube * np.cos(theta_bot),
                    R_tube * np.sin(theta_bot), "w--", lw=0.8, alpha=0.6)

            is_transient = self.page_welcome.is_transient
            status = ""
            if is_transient and r.converged:
                status = f"  [converged t={r.times_arr[-1]:.2f}s]"
            ax.set_aspect("equal")
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_title(f"Final Temperature{status} — hover for T value")
            ax.grid(True, alpha=0.2)
            pad = 0.00015
            ax.set_xlim(-pad, geom.W + pad)
            ax.set_ylim(-geom.R_tube - pad, geom.H + pad)

            # Attach hover interpolator
            self.page_results.plot.set_interpolator(triang, r.T_full)
            self.page_results.plot.fig.tight_layout()
            self.page_results.plot.canvas.draw()

    def _restart(self):
        self.mesh_result = None
        self.solve_result = None
        self._update_page(0)

    def _save_plots(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder")
        if not folder:
            return
        try:
            # Save solve page plot
            path1 = os.path.join(folder, "solution.png")
            self.page_solve.plot.fig.savefig(path1, dpi=300, bbox_inches="tight")
            self.page_results.console.log(f"Saved: {path1}")

            # Save results page plot
            path2 = os.path.join(folder, "results.png")
            self.page_results.plot.fig.savefig(path2, dpi=300, bbox_inches="tight")
            self.page_results.console.log(f"Saved: {path2}")

            # Save mesh plot
            path3 = os.path.join(folder, "mesh.png")
            self.page_mesh.plot.fig.savefig(path3, dpi=300, bbox_inches="tight")
            self.page_results.console.log(f"Saved: {path3}")

            QMessageBox.information(self, "Saved", f"Plots saved to:\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _view_3d_mesh(self):
        """Open PyVista window showing 3D mesh with material coloring."""
        mesh = self.mesh_result
        if mesh is None:
            return
        try:
            import pyvista as pv
            Pmm = mesh.coords * 1e3

            def _make_grid(tets):
                cells = np.hstack([
                    np.full((tets.shape[0], 1), 4, dtype=np.int64), tets
                ]).ravel()
                ct = np.full(tets.shape[0], pv.CellType.TETRA, dtype=np.uint8)
                return pv.UnstructuredGrid(cells, ct, Pmm)

            grid_tube  = _make_grid(mesh.tets_tube)
            grid_paste = _make_grid(mesh.tets_paste)
            grid_mold  = _make_grid(mesh.tets_mold)

            surf_tube  = grid_tube.extract_surface(algorithm=None).triangulate().clean()
            surf_paste = grid_paste.extract_surface(algorithm=None).triangulate().clean()
            surf_mold  = grid_mold.extract_surface(algorithm=None).triangulate().clean()

            pl = pv.Plotter()
            pl.add_text(
                f"3D Mesh — {mesh.n_nodes:,} nodes  {mesh.n_elements:,} tets\n"
                f"Tube: {len(mesh.tets_tube):,}  Paste: {len(mesh.tets_paste):,}  "
                f"Mold: {len(mesh.tets_mold):,}",
                font_size=9)
            pl.add_mesh(surf_tube,  color="#EF5350", opacity=0.85,
                        show_edges=True, edge_color="#333333", label="Tube")
            pl.add_mesh(surf_paste, color="#42A5F5", opacity=0.70,
                        show_edges=True, edge_color="#333333", label="Paste")
            pl.add_mesh(surf_mold,  color="#66BB6A", opacity=0.55,
                        show_edges=True, edge_color="#333333", label="Mold")
            pl.add_legend([["Tube", "#EF5350"], ["Paste", "#42A5F5"],
                           ["Mold", "#66BB6A"]])
            pl.view_isometric()
            pl.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
            pl.show()
        except ImportError:
            QMessageBox.warning(self, "PyVista Missing",
                                "Install pyvista: pip install pyvista")
        except Exception as e:
            QMessageBox.critical(self, "Mesh View Error", str(e))

    def _view_3d(self):
        """Open full interactive PyVista slicer: body colored by T + X/Y/Z clip planes + hover."""
        if self.solve_result is None:
            return
        r = self.solve_result
        geom, _, bc, _, _ = self.page_params.get_params()
        try:
            import pyvista as pv

            Pmm = r.coords_full * 1e3
            cells = np.hstack([
                np.full((r.tets_full.shape[0], 1), 4, dtype=np.int64),
                r.tets_full
            ]).ravel()
            celltypes = np.full(r.tets_full.shape[0], pv.CellType.TETRA,
                                dtype=np.uint8)
            grid = pv.UnstructuredGrid(cells, celltypes, Pmm)
            grid.point_data["T"]  = r.T_full.astype(float)
            grid.point_data["dT"] = (r.T_full - bc.Tinf).astype(float)

            def _robust_clim(vals, plo=1.0, phi=99.0):
                lo, hi = float(np.percentile(vals, plo)), float(np.percentile(vals, phi))
                span = hi - lo
                if not np.isfinite(span) or span < 1e-6:
                    mid = 0.5 * (lo + hi)
                    lo, hi = mid - 0.5e-6, mid + 0.5e-6
                    span = hi - lo
                pad = 0.02 * span
                return (lo - pad, hi + pad)

            # Build initial clean surface for clim reference
            surf_ref = grid.extract_surface(algorithm=None).clean()
            T_CLIM = _robust_clim(np.asarray(surf_ref.point_data["T"], float))
            dT_vals = np.asarray(grid.point_data["dT"], float)
            DT_CLIM = _robust_clim(dT_vals)

            xmin, xmax = float(grid.points[:, 0].min()), float(grid.points[:, 0].max())
            ymin, ymax = float(grid.points[:, 1].min()), float(grid.points[:, 1].max())
            zmin, zmax = float(grid.points[:, 2].min()), float(grid.points[:, 2].max())
            x0 = 0.5 * (xmin + xmax)
            y0 = 0.5 * (ymin + ymax)
            z0 = 0.5 * (zmin + zmax)

            cut = {"x": xmax, "y": ymax, "z": zmax}  # start: no clipping
            actors = {"body": None, "sx": None, "sy": None, "sz": None}

            p = pv.Plotter()
            p.add_text("T distribution — drag sliders to clip  |  right-click to pick temperature",
                       font_size=9)

            def _redraw():
                for k, a in actors.items():
                    if a is not None:
                        p.remove_actor(a)

                vol = grid
                if cut["x"] < xmax - 1e-6:
                    vol = vol.clip(normal=(1, 0, 0),
                                   origin=(cut["x"], 0.0, 0.0), invert=True)
                if cut["y"] < ymax - 1e-6:
                    vol = vol.clip(normal=(0, 1, 0),
                                   origin=(0.0, cut["y"], 0.0), invert=True)
                if cut["z"] < zmax - 1e-6:
                    vol = vol.clip(normal=(0, 0, 1),
                                   origin=(0.0, 0.0, cut["z"]), invert=True)

                body = vol.extract_surface(algorithm=None).clean()
                if body.n_points > 0:
                    Tb = np.asarray(body.point_data["T"], float)
                    clim_b = _robust_clim(Tb)
                else:
                    clim_b = T_CLIM

                actors["body"] = p.add_mesh(
                    body, scalars="T", preference="point",
                    cmap="turbo", clim=clim_b, show_edges=False,
                    name="body",
                    scalar_bar_args={"title": "T (°C)", "fmt": "%.3g"})

                slx = grid.slice(normal=(1, 0, 0), origin=(cut["x"], 0.0, 0.0))
                sly = grid.slice(normal=(0, 1, 0), origin=(0.0, cut["y"], 0.0))
                slz = grid.slice(normal=(0, 0, 1), origin=(0.0, 0.0, cut["z"]))

                dT_clim = _robust_clim(
                    np.asarray(slx.point_data["dT"], float)
                    if slx.n_points > 0 else dT_vals)

                for key, sl, show_bar in [("sx", slx, True),
                                           ("sy", sly, False),
                                           ("sz", slz, False)]:
                    if sl.n_points > 0:
                        actors[key] = p.add_mesh(
                            sl, scalars="dT", preference="point",
                            cmap="turbo", clim=dT_clim, show_edges=False,
                            name=key, reset_camera=False,
                            show_scalar_bar=show_bar,
                            scalar_bar_args={"title": "ΔT (°C)", "fmt": "%.3g"})

            _redraw()

            p.add_slider_widget(
                lambda v: (cut.update({"x": float(v)}), _redraw()),
                rng=[xmin, xmax], value=xmax,
                title="Cut X (mm)", pointa=(0.03, 0.12), pointb=(0.35, 0.12),
                style="modern")
            p.add_slider_widget(
                lambda v: (cut.update({"y": float(v)}), _redraw()),
                rng=[ymin, ymax], value=ymax,
                title="Cut Y (mm)", pointa=(0.03, 0.07), pointb=(0.35, 0.07),
                style="modern")
            p.add_slider_widget(
                lambda v: (cut.update({"z": float(v)}), _redraw()),
                rng=[zmin, zmax], value=zmax,
                title="Cut Z (mm)", pointa=(0.03, 0.02), pointb=(0.35, 0.02),
                style="modern")

            # Right-click pick: show temperature at clicked point
            _pick_label = [None]
            def _on_pick(point):
                if _pick_label[0] is not None:
                    p.remove_actor(_pick_label[0])
                # find nearest node
                idx = int(np.argmin(np.sum((r.coords_full * 1e3 - point)**2, axis=1)))
                T_val = r.T_full[idx]
                label_pt = r.coords_full[idx] * 1e3
                _pick_label[0] = p.add_point_labels(
                    [label_pt], [f"T = {T_val:.4f} °C"],
                    font_size=12, text_color="yellow",
                    point_color="yellow", point_size=10,
                    name="pick_label", reset_camera=False)

            p.enable_point_picking(callback=_on_pick, show_message=False,
                                   picker="point", use_picker=True,
                                   left_clicking=False)

            p.view_isometric()
            p.camera.zoom(1.2)
            p.show_grid(xtitle="X (mm)", ytitle="Y (mm)", ztitle="Z (mm)")
            p.show()

        except ImportError:
            QMessageBox.warning(self, "PyVista Missing",
                                "PyVista is required for 3D visualization.")
        except Exception as e:
            QMessageBox.critical(self, "3D View Error", str(e))

    def _generate_gif(self):
        r = self.solve_result
        if r is None or not hasattr(r, '_make_full_T'):
            QMessageBox.warning(self, "No Data", "Transient solution needed for GIF.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Save GIF to folder")
        if not folder:
            return

        if self.page_welcome.is_3d:
            self._generate_gif_3d(folder, r)
        else:
            self._generate_gif_2d(folder, r)

    def _render_frames_to_gif(self, frames, folder, filename, duration, log):
        import io
        from PIL import Image
        gif_path = os.path.join(folder, filename)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=duration, loop=0, optimize=False)
        log(f"GIF saved: {gif_path}  ({len(frames)} frames)")
        QMessageBox.information(None, "GIF Saved",
                                f"Animation saved!\n{gif_path}\n{len(frames)} frames")

    def _generate_gif_2d(self, folder, r):
        self.page_results.console.log("Generating 2D animation GIF ...")
        QApplication.processEvents()
        try:
            import io
            from PIL import Image
            import matplotlib
            import matplotlib.pyplot as plt

            prev_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

            geom, _, bc, _, _ = self.page_params.get_params()
            triang = mtri.Triangulation(r.coords_full[:, 0], r.coords_full[:, 1],
                                        r.tris_full)
            vals_all = [r._make_full_T(tf) for tf in r.T_history]
            n_levels = 50
            xc = geom.xc
            R_tube, R_paste = geom.R_tube, geom.R_paste
            W, H = geom.W, geom.H
            pad = 0.00015
            theta_top = np.linspace(0, np.pi, 400)
            theta_bot = np.linspace(-np.pi, 0, 400)
            n_frames = len(vals_all)

            # Global range for reference (shown in title)
            T_all = np.concatenate([v.ravel() for v in vals_all])
            T_gmin, T_gmax = float(T_all.min()), float(T_all.max())

            pil_frames = []
            for i, Z in enumerate(vals_all):
                fig, ax = plt.subplots(figsize=(7.0, 6.0), dpi=120)
                fig.patch.set_facecolor("#1e1e1e")
                ax.set_facecolor("#1e1e1e")

                # Per-frame normalization — prevents single-color isothermal frames
                zf_min = float(np.min(Z))
                zf_max = float(np.max(Z))
                span = zf_max - zf_min
                if span < 0.05:
                    mid = 0.5 * (zf_min + zf_max)
                    zf_min, zf_max = mid - 0.5, mid + 0.5
                    span = zf_max - zf_min
                lvl_frame = np.linspace(zf_min, zf_max, n_levels)

                cf = ax.tricontourf(triang, Z, levels=lvl_frame, cmap="turbo")
                cbar = fig.colorbar(cf, ax=ax, pad=0.03)
                cbar.set_label("T (°C)", color="#ccc")
                cbar.ax.yaxis.set_tick_params(color="#ccc")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc")
                for rv in [R_tube, R_paste]:
                    ax.plot(xc + rv * np.cos(theta_top),
                            rv * np.sin(theta_top), "w--", lw=0.8, alpha=0.7)
                ax.plot(xc + R_tube * np.cos(theta_bot),
                        R_tube * np.sin(theta_bot), "w--", lw=0.8, alpha=0.7)
                ax.set_aspect("equal")
                ax.set_xlim(-pad, W + pad)
                ax.set_ylim(-R_tube - pad, H + pad)
                ax.set_xlabel("x (m)", color="#ccc")
                ax.set_ylabel("y (m)", color="#ccc")
                ax.tick_params(colors="#ccc")
                ax.set_title(
                    f"t = {r.times_anim[i]:.3f} s  [{i+1}/{n_frames}]   "
                    f"T: {zf_min:.2f}–{zf_max:.2f} °C  (global: {T_gmin:.1f}–{T_gmax:.1f})",
                    color="#ddd", fontsize=9)
                ax.grid(True, alpha=0.2, color="#444444")
                for sp_ in ax.spines.values():
                    sp_.set_color("#555555")
                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
                buf.seek(0)
                pil_frames.append(Image.open(buf).copy())
                buf.close()
                plt.close(fig)
                if (i + 1) % 5 == 0 or i == n_frames - 1:
                    self.page_results.console.log(f"  Frame {i+1}/{n_frames}")
                    QApplication.processEvents()

            matplotlib.use(prev_backend)
            self._render_frames_to_gif(pil_frames, folder, "transient_2d.gif", 120,
                                       self.page_results.console.log)
        except Exception as e:
            import traceback
            self.page_results.console.log(f"GIF Error: {traceback.format_exc()}")
            QMessageBox.critical(self, "GIF Error", str(e))

    def _generate_gif_3d(self, folder, r):
        """GIF for 3D transient: z=0 cross-section tricontourf frames
           + side-by-side convergence plot."""
        self.page_results.console.log("Building 3D animation GIF (z=0 cross-section) ...")
        QApplication.processEvents()
        try:
            import io
            from PIL import Image
            import matplotlib
            import matplotlib.pyplot as plt
            from scipy.spatial import Delaunay

            prev_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

            geom, _, bc, _, _ = self.page_params.get_params()
            mesh = self.mesh_result
            P_half = mesh.coords          # (Nn, 3) half-domain

            # --- Build z=0 cross-section triangulation ---
            # Include BOTH Robin (outer) and Dirichlet (inner tube) faces at z=0
            all_bc_tris = np.vstack([mesh.tri_rob, mesh.tri_dir]) if len(mesh.tri_dir) > 0 else mesh.tri_rob
            Pc_all = P_half[all_bc_tris].mean(axis=1)
            z_tol = P_half[:, 2].max() * 0.02 + 1e-8
            z0_mask = np.isclose(Pc_all[:, 2], 0.0, atol=z_tol)
            tri_z0 = all_bc_tris[z0_mask]
            if len(tri_z0) == 0:
                z0_mask = Pc_all[:, 2] < P_half[:, 2].max() * 0.1
                tri_z0 = all_bc_tris[z0_mask]
            z0_nodes = np.unique(tri_z0)
            node_remap = np.full(P_half.shape[0], -1, dtype=int)
            node_remap[z0_nodes] = np.arange(len(z0_nodes))
            tri_z0_local = node_remap[tri_z0]
            # mirror the z=0 cross-section
            pts_x_half = P_half[z0_nodes, 0] * 1e3   # mm
            pts_y_half = P_half[z0_nodes, 1] * 1e3
            xc_mm = geom.xc * 1e3
            mirror_x = 2.0 * xc_mm - pts_x_half
            all_x = np.concatenate([pts_x_half, mirror_x])
            all_y = np.concatenate([pts_y_half, pts_y_half])
            n_half = len(z0_nodes)
            tri_mirror = np.column_stack([
                tri_z0_local[:, 0] + n_half,
                tri_z0_local[:, 2] + n_half,
                tri_z0_local[:, 1] + n_half,
            ])
            tri_full_local = np.vstack([tri_z0_local, tri_mirror])
            triang = mtri.Triangulation(all_x, all_y, tri_full_local)

            # Global range for reference in title
            T_all_frames = np.array([r._make_full_T(tf)[z0_nodes]
                                     for tf in r.T_history])
            T_gmin = float(T_all_frames.min())
            T_gmax = float(T_all_frames.max())

            R_tube_mm = geom.R_tube * 1e3
            R_paste_mm = geom.R_paste * 1e3
            W_mm = geom.W_mm
            H_mm = geom.H_mm
            theta_top = np.linspace(0, np.pi, 400)
            theta_bot = np.linspace(-np.pi, 0, 400)
            n_frames = len(r.T_history)
            n_levels = 50

            # Precompute step indices for GIF frames
            has_dT = r.dT_hist is not None and len(r.dT_hist) > 0
            n_total_steps = len(r.tmax_hist)

            pil_frames = []
            for i, T_free in enumerate(r.T_history):
                T_half_nodes = r._make_full_T(T_free)[z0_nodes]
                T_full_nodes = np.concatenate([T_half_nodes, T_half_nodes])

                fig, (ax_T, ax_conv) = plt.subplots(
                    1, 2, figsize=(12.0, 5.5), dpi=110,
                    gridspec_kw={"width_ratios": [1.2, 1]})
                fig.patch.set_facecolor("#1e1e1e")

                # --- Left: spatial cross-section (per-frame normalization) ---
                ax_T.set_facecolor("#1e1e1e")
                zf_min = float(np.min(T_half_nodes))
                zf_max = float(np.max(T_half_nodes))
                if zf_max - zf_min < 0.05:
                    mid = 0.5 * (zf_min + zf_max)
                    zf_min, zf_max = mid - 0.5, mid + 0.5
                lvl_frame = np.linspace(zf_min, zf_max, n_levels)

                cf = ax_T.tricontourf(triang, T_full_nodes, levels=lvl_frame, cmap="turbo")
                cbar = fig.colorbar(cf, ax=ax_T, pad=0.03)
                cbar.set_label("T (°C)", color="#ccc")
                cbar.ax.yaxis.set_tick_params(color="#ccc")
                plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#ccc")
                for rv in [R_tube_mm, R_paste_mm]:
                    ax_T.plot(xc_mm + rv * np.cos(theta_top),
                              rv * np.sin(theta_top), "w--", lw=0.8, alpha=0.7)
                ax_T.plot(xc_mm + R_tube_mm * np.cos(theta_bot),
                          R_tube_mm * np.sin(theta_bot), "w--", lw=0.8, alpha=0.7)
                ax_T.set_aspect("equal")
                ax_T.set_xlim(0, W_mm)
                ax_T.set_ylim(-R_tube_mm - 0.05, H_mm + 0.05)
                ax_T.set_xlabel("x (mm)", color="#ccc")
                ax_T.set_ylabel("y (mm)", color="#ccc")
                ax_T.tick_params(colors="#ccc")
                ax_T.set_title(
                    f"z=0  t={r.times_anim[i]:.3f}s  [{i+1}/{n_frames}]\n"
                    f"T: {zf_min:.2f}–{zf_max:.2f}°C  (global {T_gmin:.1f}–{T_gmax:.1f}°C)",
                    color="#ddd", fontsize=8)
                ax_T.grid(True, alpha=0.15, color="#444444")
                for sp_ in ax_T.spines.values():
                    sp_.set_color("#555555")

                # --- Right: ANSYS-style convergence (residual log + Tmax) ---
                ax_conv.set_facecolor("#1e1e1e")
                # current step index for this frame
                cur_step = min(
                    np.searchsorted(r.times_arr, r.times_anim[i], side="right"),
                    n_total_steps)
                steps_shown = np.arange(cur_step)

                if has_dT and cur_step > 1:
                    dT_shown = r.dT_hist[:min(cur_step - 1, len(r.dT_hist))]
                    steps_dT = np.arange(1, len(dT_shown) + 1)
                    ax_conv.semilogy(steps_dT, np.maximum(dT_shown, 1e-12),
                                     color="#ff9800", lw=1.5, label="max|ΔT|")
                    _, sp_obj = self.page_params.get_params()[0], self.page_params.get_params()[4]
                    ax_conv.axhline(sp_obj.conv_tol, color="#ff5252", ls="--",
                                    alpha=0.8, label=f"tol={sp_obj.conv_tol:.0e}")
                    ax_conv.set_ylabel("max|ΔT| (°C)", color="#ccc")
                    ax_conv.set_title("Residual Convergence", color="#ddd")
                else:
                    T_shown = r.tmax_hist[:cur_step]
                    ax_conv.plot(steps_shown[:len(T_shown)], T_shown,
                                 color="#69f0ae", lw=1.5, label="Tmax")
                    ax_conv.axhline(bc.T_wall, color="#ef5350", ls="--",
                                    alpha=0.7, label=f"T_wall={bc.T_wall}°C")
                    ax_conv.set_ylabel("Tmax (°C)", color="#ccc")
                    ax_conv.set_title("Convergence", color="#ddd")

                ax_conv.set_xlabel("Time Step", color="#ccc")
                ax_conv.tick_params(colors="#ccc")
                ax_conv.legend(loc="upper right", fontsize=8,
                               facecolor="#333333", labelcolor="#ccc")
                ax_conv.grid(True, alpha=0.25, color="#444444", which="both")
                for sp_ in ax_conv.spines.values():
                    sp_.set_color("#555555")

                fig.suptitle(f"3D Transient — frame {i+1}/{n_frames}",
                             color="#aaa", fontsize=10)
                fig.tight_layout()

                buf = io.BytesIO()
                fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
                buf.seek(0)
                pil_frames.append(Image.open(buf).copy())
                buf.close()
                plt.close(fig)

                if (i + 1) % 5 == 0 or i == n_frames - 1:
                    self.page_results.console.log(f"  Frame {i+1}/{n_frames}")
                    QApplication.processEvents()

            matplotlib.use(prev_backend)
            self._render_frames_to_gif(pil_frames, folder, "transient_3d.gif", 150,
                                       self.page_results.console.log)
        except Exception as e:
            import traceback
            self.page_results.console.log(f"GIF Error: {traceback.format_exc()}")
            QMessageBox.critical(self, "GIF Error", str(e))

    def _pin_run(self):
        if self.solve_result is None:
            return
        geom, mat, bc, _, sp = self.page_params.get_params()
        solver_type = self.page_welcome.solver_type
        label = (f"{solver_type.upper()} | k_tube={mat.k_tube} | "
                 f"k_paste={mat.k_paste} | bc_inner={bc.bc_inner}")
        self.page_results._pinned_runs.append((label, self.solve_result, bc, mat))
        n = len(self.page_results._pinned_runs)
        self.page_results.btn_compare.setEnabled(n >= 2)
        self.page_results.console.log(
            f"Pinned run #{n}: {label}")
        QMessageBox.information(self, "Run Pinned",
                                f"Run #{n} pinned.\n{label}\n\n"
                                f"Pin another run to enable comparison.")

    def _compare_runs(self):
        runs = self.page_results._pinned_runs
        if len(runs) < 2:
            return
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            prev_backend = matplotlib.get_backend()
            matplotlib.use("QtAgg")

            n = len(runs)
            colors = ["#69f0ae", "#ff9800", "#42a5f5", "#ef5350",
                      "#ce93d8", "#fff176"]

            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.patch.set_facecolor("#1e1e1e")

            labels = [r[0] for r in runs]
            short_labels = [f"Run {i+1}" for i in range(n)]

            # Panel 1: T_min / T_max bar chart
            ax1 = axes[0]
            ax1.set_facecolor("#1e1e1e")
            x = np.arange(n)
            T_mins = [r[1].T_min for r in runs]
            T_maxs = [r[1].T_max for r in runs]
            ax1.bar(x - 0.2, T_mins, 0.35,
                    color=[c + "99" for c in colors[:n]], label="T_min")
            ax1.bar(x + 0.2, T_maxs, 0.35,
                    color=colors[:n], label="T_max")
            ax1.set_xticks(x)
            ax1.set_xticklabels(short_labels, color="#ccc")
            ax1.set_ylabel("Temperature (°C)", color="#ccc")
            ax1.set_title("T_min / T_max Comparison", color="#ddd")
            ax1.tick_params(colors="#ccc")
            ax1.legend(facecolor="#333333", labelcolor="#ccc")
            ax1.grid(True, alpha=0.2, axis="y")
            for sp_ in ax1.spines.values():
                sp_.set_color("#555555")

            # Panel 2: Tmax convergence overlay (transient only)
            ax2 = axes[1]
            ax2.set_facecolor("#1e1e1e")
            has_transient = False
            for i, (lbl, r, bc, mat) in enumerate(runs):
                if r.tmax_hist is not None and r.times_arr is not None:
                    steps = np.arange(len(r.tmax_hist))
                    ax2.plot(steps, r.tmax_hist, color=colors[i % len(colors)],
                             lw=1.5, label=short_labels[i])
                    has_transient = True
            if not has_transient:
                ax2.text(0.5, 0.5, "Transient data\nnot available",
                         transform=ax2.transAxes, ha="center", va="center",
                         color="#888", fontsize=12)
            ax2.set_xlabel("Time Step", color="#ccc")
            ax2.set_ylabel("Tmax (°C)", color="#ccc")
            ax2.set_title("Convergence Overlay", color="#ddd")
            ax2.tick_params(colors="#ccc")
            ax2.legend(facecolor="#333333", labelcolor="#ccc")
            ax2.grid(True, alpha=0.25)
            for sp_ in ax2.spines.values():
                sp_.set_color("#555555")

            # Panel 3: Summary table
            ax3 = axes[2]
            ax3.set_facecolor("#1e1e1e")
            ax3.set_axis_off()
            rows = []
            for i, (lbl, r, bc, mat) in enumerate(runs):
                conv_str = (f"step {r.conv_step}" if r.converged else "—")
                rows.append([short_labels[i],
                             f"{r.T_min:.2f}",
                             f"{r.T_max:.2f}",
                             f"{mat.k_paste:.1f}",
                             conv_str])
            col_labels = ["Run", "T_min°C", "T_max°C", "k_paste", "Conv"]
            tbl = ax3.table(cellText=rows, colLabels=col_labels,
                            loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(9)
            for (row, col), cell in tbl.get_celld().items():
                cell.set_facecolor("#2a2a2a" if row > 0 else "#333333")
                cell.set_text_props(color="#ccc")
                cell.set_edgecolor("#555555")
            ax3.set_title("Summary Table", color="#ddd", pad=60)

            # Legend below
            legend_text = "\n".join(
                [f"{short_labels[i]}: {runs[i][0][:60]}" for i in range(n)])
            fig.text(0.5, 0.01, legend_text, ha="center", va="bottom",
                     color="#888", fontsize=7, wrap=True)

            fig.suptitle("Run Comparison", color="#ccc", fontsize=13)
            fig.tight_layout(rect=[0, 0.08, 1, 0.95])
            plt.show()
            matplotlib.use(prev_backend)
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Compare Error", traceback.format_exc())


# =====================================================================
# ENTRY POINT
# =====================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # ---- Dark mode palette ----
    palette = QPalette()
    dark     = QColor(30, 30, 30)
    mid_dark = QColor(42, 42, 42)
    mid      = QColor(55, 55, 55)
    light_g  = QColor(200, 200, 200)
    white    = QColor(220, 220, 220)
    accent   = QColor(42, 130, 218)

    palette.setColor(QPalette.ColorRole.Window, mid_dark)
    palette.setColor(QPalette.ColorRole.WindowText, white)
    palette.setColor(QPalette.ColorRole.Base, dark)
    palette.setColor(QPalette.ColorRole.AlternateBase, mid_dark)
    palette.setColor(QPalette.ColorRole.ToolTipBase, mid)
    palette.setColor(QPalette.ColorRole.ToolTipText, white)
    palette.setColor(QPalette.ColorRole.Text, white)
    palette.setColor(QPalette.ColorRole.Button, mid)
    palette.setColor(QPalette.ColorRole.ButtonText, white)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 50, 50))
    palette.setColor(QPalette.ColorRole.Link, accent)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(140, 140, 140))
    app.setPalette(palette)

    # Global dark stylesheet for widgets that palette alone doesn't cover
    app.setStyleSheet("""
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 14px;
            color: #ddd;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 2px 4px;
        }
        QPlainTextEdit {
            background-color: #1a1a1a;
            color: #00ff88;
            border: 1px solid #444;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        QProgressBar::chunk {
            background-color: #2a82da;
            border-radius: 2px;
        }
        QRadioButton { color: #ddd; }
        QLabel { color: #ddd; }
        QMessageBox { background-color: #2a2a2a; }
        QMessageBox QLabel { color: #ddd; }
    """)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
