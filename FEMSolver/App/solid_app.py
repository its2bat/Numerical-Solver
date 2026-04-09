"""
Solid Mechanics Beam Analysis — PyQt6 wizard window.

Flow: Select beam type → Parameters → Mesh & Solve → Results
"""

import sys
import os
import traceback
import numpy as np

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PyQt6.QtWidgets import (
    QWidget, QMainWindow, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup,
    QGroupBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit,
    QProgressBar, QStackedWidget, QSplitter, QFrame,
    QComboBox, QSizePolicy, QMessageBox, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from solid.materials import MATERIALS
from solid.beam_mesh import mesh_straight_cantilever, mesh_zigzag_cantilever
from solid.beam_solver import (assemble_plane_stress, tip_load_vector,
                                solve_beam, tip_displacement, spring_constant,
                                von_mises)
from solid.beam_postprocess import (
    analytical_straight, analytical_zigzag,
    displacement_magnitude, plot_beam_results, result_summary
)


# ─────────────────────────────────────────────────────────────────────────────
# SHARED WIDGETS
# ─────────────────────────────────────────────────────────────────────────────

_DARK_BG  = "#2a2a2a"
_DARK_AX  = "#1e1e1e"
_ACCENT   = "#64B5F6"
_ACCENT2  = "#FF9800"
_FG       = "#cccccc"

_BASE_STYLE = f"""
QWidget        {{ background-color: #2a2a2a; color: #cccccc; }}
QGroupBox      {{ border: 1px solid #555; border-radius:4px;
                  margin-top: 10px; padding-top: 6px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; color: #aaa; }}
QDoubleSpinBox, QSpinBox, QComboBox
               {{ background:#3a3a3a; color:#ccc; border:1px solid #555;
                  border-radius:3px; padding:2px 6px; }}
QPlainTextEdit {{ background:#1a1a1a; color:#bbb; border:1px solid #444; }}
QProgressBar   {{ border:1px solid #555; border-radius:3px;
                  text-align:center; }}
QProgressBar::chunk {{ background:#1565C0; }}
QCheckBox      {{ color:#ccc; }}
"""


def _btn(text, color="#1565C0", hover=None):
    hov = hover or color
    s = (f"QPushButton {{ background-color:{color}; color:white; "
         f"border-radius:5px; padding:8px 18px; font-weight:bold; }}"
         f"QPushButton:hover {{ background-color:{hov}; }}"
         f"QPushButton:disabled {{ background-color:#555; color:#888; }}")
    b = QPushButton(text)
    b.setStyleSheet(s)
    b.setMinimumHeight(34)
    return b


class LogConsole(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 9))
        self.setMaximumBlockCount(2000)

    def log(self, txt):
        self.appendPlainText(txt)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def clear_log(self):
        self.clear()


class MplPanel(QWidget):
    """Embedded matplotlib canvas with toolbar."""
    def __init__(self, width=12, height=5.5):
        super().__init__()
        self.fig = Figure(figsize=(width, height), dpi=100, facecolor=_DARK_BG)
        self.canvas   = FigureCanvasQTAgg(self.fig)
        self.toolbar  = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)


class StepIndicator(QWidget):
    STEPS = ["Beam Type", "Parameters", "Mesh & Solve", "Results"]

    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self._labels = []
        for i, name in enumerate(self.STEPS):
            lbl = QLabel(f"  {i+1}. {name}  ")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(lbl)
            self._labels.append(lbl)
            if i < len(self.STEPS) - 1:
                arr = QLabel("→")
                arr.setStyleSheet("color:#666; font-weight:bold;")
                layout.addWidget(arr)

    def set_active(self, idx):
        for i, lbl in enumerate(self._labels):
            if i < idx:
                lbl.setStyleSheet(
                    "padding:4px 8px; border-radius:4px; "
                    "background:#2E7D32; color:white; font-weight:bold;")
            elif i == idx:
                lbl.setStyleSheet(
                    "padding:4px 8px; border-radius:4px; "
                    "background:#1565C0; color:white; font-weight:bold;")
            else:
                lbl.setStyleSheet(
                    "padding:4px 8px; border-radius:4px; "
                    "background:#444; color:#999;")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 0 — BEAM TYPE SELECTION
# ─────────────────────────────────────────────────────────────────────────────

class BeamTypePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        title = QLabel("Select Beam Geometry")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{_ACCENT}; margin:12px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Cards
        card_row = QHBoxLayout()

        self._grp = QButtonGroup(self)

        for tag, heading, body in [
            ("straight",
             "Straight Cantilever",
             "Rectangular beam, fixed at one end.\n"
             "Transverse tip load in y-direction.\n\n"
             "Analytical: k = 3EI/L³\n"
             "FEM vs Euler-Bernoulli comparison."),
            ("zigzag",
             "Zig-Zag Cantilever",
             "Serpentine (meander) beam with n segments.\n"
             "Used in MEMS springs & accelerometers.\n\n"
             "Analytical: free-end & guided-end bounds.\n"
             "Configurable gap g and segment count."),
        ]:
            card = QGroupBox()
            card.setMinimumWidth(260)
            cv = QVBoxLayout(card)
            rb = QRadioButton(heading)
            rb.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
            rb.setStyleSheet(f"color:{_ACCENT};")
            rb.setProperty("tag", tag)
            self._grp.addButton(rb)
            cv.addWidget(rb)
            desc = QLabel(body)
            desc.setWordWrap(True)
            desc.setStyleSheet("color:#aaa; font-size:10pt;")
            cv.addWidget(desc)
            card_row.addWidget(card)

        self._grp.buttons()[0].setChecked(True)
        layout.addLayout(card_row)
        layout.addStretch()

    @property
    def beam_type(self):
        for btn in self._grp.buttons():
            if btn.isChecked():
                return btn.property("tag")
        return "straight"


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────

class _DSpin(QDoubleSpinBox):
    """Convenient double spin with large range."""
    def __init__(self, val, lo=0.0, hi=1e9, dec=4, suffix=""):
        super().__init__()
        self.setRange(lo, hi)
        self.setDecimals(dec)
        self.setValue(val)
        self.setSuffix(suffix)
        self.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)


class ParametersPage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("Beam Parameters")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{_ACCENT};")
        layout.addWidget(title)

        # ── material preset ──────────────────────────────────────────────
        mat_grp = QGroupBox("Material")
        mf = QFormLayout(mat_grp)
        self.combo_mat = QComboBox()
        self.combo_mat.addItems(list(MATERIALS.keys()))
        mf.addRow("Preset:", self.combo_mat)
        layout.addWidget(mat_grp)

        # ── geometry ─────────────────────────────────────────────────────
        geo_grp = QGroupBox("Geometry  (values in µm, converted to m internally)")
        gf = QFormLayout(geo_grp)

        self.sp_L  = _DSpin(200.0, 1.0, 1e7, 2, " µm")
        self.sp_W  = _DSpin(5.0,   0.1, 1e6, 2, " µm")
        self.sp_t  = _DSpin(2.0,   0.1, 1e6, 2, " µm")
        gf.addRow("Length L:", self.sp_L)
        gf.addRow("Width W:",  self.sp_W)
        gf.addRow("Thickness t (out-of-plane):", self.sp_t)

        # Zig-zag extras (shown/hidden dynamically)
        self._zz_labels = []
        self._zz_spins  = []
        for label, spin in [
            ("Gap g:",            _DSpin(4.0, 0.1, 1e5, 2, " µm")),
            ("Segments n:",       None),
        ]:
            if spin is None:
                spin = QSpinBox()
                spin.setRange(1, 50)
                spin.setValue(4)
            lbl = QLabel(label)
            gf.addRow(lbl, spin)
            self._zz_labels.append(lbl)
            self._zz_spins.append(spin)
        self.sp_g      = self._zz_spins[0]
        self.sp_n_segs = self._zz_spins[1]

        layout.addWidget(geo_grp)

        # ── material props ────────────────────────────────────────────────
        prop_grp = QGroupBox("Material Properties")
        pf = QFormLayout(prop_grp)
        self.sp_E  = _DSpin(170.0, 0.001, 1e6, 3, " GPa")
        self.sp_nu = _DSpin(0.28,  0.0,   0.499, 3)
        pf.addRow("Young's modulus E:", self.sp_E)
        pf.addRow("Poisson's ratio ν:", self.sp_nu)
        layout.addWidget(prop_grp)

        # ── loading ───────────────────────────────────────────────────────
        load_grp = QGroupBox("Loading")
        lf = QFormLayout(load_grp)
        self.sp_F  = _DSpin(-1e-6, -1e3, 1e3, 6, " N")
        lf.addRow("Tip force F (y-direction, - = down):", self.sp_F)
        layout.addWidget(load_grp)

        # ── mesh density ──────────────────────────────────────────────────
        mesh_grp = QGroupBox("Mesh Resolution")
        mf2 = QFormLayout(mesh_grp)
        self.sp_ny = QSpinBox(); self.sp_ny.setRange(2, 60); self.sp_ny.setValue(10)
        mf2.addRow("Cells across width (ny):", self.sp_ny)
        ny_note = QLabel("nx computed automatically for ~1:1 cell aspect ratio\n"
                         "(prevents shear locking in bending problems)")
        ny_note.setStyleSheet("color:#777; font-size:9pt;")
        ny_note.setWordWrap(True)
        mf2.addRow("", ny_note)
        layout.addWidget(mesh_grp)

        layout.addStretch()

        # ── wire-up preset ────────────────────────────────────────────────
        self.combo_mat.currentTextChanged.connect(self._apply_preset)
        self._apply_preset(self.combo_mat.currentText())

    def _apply_preset(self, name):
        data = MATERIALS.get(name, {})
        if data:
            self.sp_E.setValue(data["E"] / 1e9)
            self.sp_nu.setValue(data["nu"])

    def show_zigzag_fields(self, visible):
        for lbl, sp in zip(self._zz_labels, self._zz_spins):
            lbl.setVisible(visible)
            sp.setVisible(visible)

    def get_params(self):
        return dict(
            L      = self.sp_L.value()  * 1e-6,
            W      = self.sp_W.value()  * 1e-6,
            t      = self.sp_t.value()  * 1e-6,
            E      = self.sp_E.value()  * 1e9,
            nu     = self.sp_nu.value(),
            F      = self.sp_F.value(),
            g      = self.sp_g.value()  * 1e-6,
            n_segs = self.sp_n_segs.value(),
            ny     = self.sp_ny.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKER
# ─────────────────────────────────────────────────────────────────────────────

class BeamWorker(QThread):
    log_signal  = pyqtSignal(str)
    finished    = pyqtSignal(object)   # result dict
    error       = pyqtSignal(str)

    def __init__(self, beam_type, p):
        super().__init__()
        self.beam_type = beam_type
        self.p = p

    def _log(self, txt):
        self.log_signal.emit(txt)

    def run(self):
        try:
            p = self.p
            log = self._log

            log("── Mesh generation ──────────────────────────────")
            if self.beam_type == "straight":
                coords, tris, fixed, tip = mesh_straight_cantilever(
                    p["L"], p["W"], p["ny"], log=log)
            else:
                coords, tris, fixed, tip = mesh_zigzag_cantilever(
                    p["L"], p["W"], p["g"], p["n_segs"],
                    p["ny"], log=log)

            if len(tip) == 0:
                raise RuntimeError(
                    "No tip nodes found — check geometry parameters.")

            log(f"  Nodes: {len(coords)},  Elements: {len(tris)}")

            log("── Stiffness assembly ───────────────────────────")
            K = assemble_plane_stress(
                tris, coords, p["E"], p["nu"], p["t"], log=log)

            log("── Load vector ──────────────────────────────────")
            f = tip_load_vector(len(coords), tip, p["F"], direction=1)
            log(f"  F = {p['F']:.3e} N distributed over {len(tip)} nodes")

            log("── Linear solve ─────────────────────────────────")
            u = solve_beam(K, f, fixed, log=log)

            log("── Post-processing ──────────────────────────────")
            delta = tip_displacement(u, tip, direction=1)
            k_fem = spring_constant(abs(p["F"]), abs(delta))
            svm   = von_mises(u, tris, coords, p["E"], p["nu"])

            log(f"  δ_tip  = {delta*1e6:.6f} µm")
            log(f"  k_FEM  = {k_fem:.6f} N/m")

            results = dict(k_fem=k_fem, delta=delta)

            if self.beam_type == "straight":
                k_th, I, formula = analytical_straight(
                    p["L"], p["W"], p["t"], p["E"])
                err = abs(k_fem - k_th) / k_th * 100
                results["k_theory"] = k_th
                results["error_pct"] = err
                results["I"]        = I
                log(f"  k_theory = {k_th:.6f} N/m   (3EI/L³)")
                log(f"  Error    = {err:.3f}%")
            else:
                k_free, k_guided, I = analytical_zigzag(
                    p["L"], p["W"], p["t"], p["E"], p["n_segs"])
                results["k_free"]   = k_free
                results["k_guided"] = k_guided
                results["I"]        = I
                in_bounds = k_free <= k_fem <= k_guided
                log(f"  k_free-end   = {k_free:.6f} N/m")
                log(f"  k_guided-end = {k_guided:.6f} N/m")
                log(f"  FEM within bounds: {'YES' if in_bounds else 'NO'}")

            log("── Done ─────────────────────────────────────────")

            self.finished.emit(dict(
                coords=coords, tris=tris, u=u, svm=svm,
                fixed=fixed, tip=tip,
                beam_type=self.beam_type, params=p, results=results
            ))

        except Exception:
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — MESH & SOLVE
# ─────────────────────────────────────────────────────────────────────────────

class MeshSolvePage(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)

        title = QLabel("Mesh & Solve")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{_ACCENT};")
        layout.addWidget(title)

        ctrl = QHBoxLayout()
        self.btn_run = _btn("Run Analysis", _ACCENT2, "#F57C00")
        ctrl.addWidget(self.btn_run)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminate
        self.progress.setVisible(False)
        ctrl.addWidget(self.progress)
        layout.addLayout(ctrl)

        splitter = QSplitter(Qt.Orientation.Vertical)
        self.plot = MplPanel(12, 4.5)
        splitter.addWidget(self.plot)
        self.console = LogConsole()
        self.console.setMaximumHeight(130)
        splitter.addWidget(self.console)
        splitter.setSizes([999, 130])
        layout.addWidget(splitter)

    def show_mesh(self, coords, tris, fixed, tip):
        """Mesh preview — aspect-ratio-aware so slender beams are readable."""
        import matplotlib.tri as mtri
        self.plot.fig.clear()
        ax = self.plot.fig.add_subplot(111)
        ax.set_facecolor(_DARK_AX)
        ax.tick_params(colors=_FG)
        for sp in ax.spines.values():
            sp.set_color("#555")

        # Fill domain with a solid colour so the shape is clear
        tri_obj = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)
        ax.tripcolor(tri_obj, np.ones(len(coords)),
                     cmap="Blues_r", vmin=0.5, vmax=1.5,
                     edgecolors="none", alpha=0.7)

        # Subsample element edges so individual cells are visible
        N = max(1, len(tris) // 800)
        tris_sub = tris[::N]
        tri_sub = mtri.Triangulation(coords[:, 0], coords[:, 1], tris_sub)
        ax.triplot(tri_sub, linewidth=0.5, color="white", alpha=0.35)

        # Fixed / tip markers (larger so they show up)
        ax.scatter(coords[fixed, 0], coords[fixed, 1],
                   s=18, c="red",   zorder=5, label=f"Fixed ({len(fixed)} nodes)")
        ax.scatter(coords[tip, 0],   coords[tip, 1],
                   s=18, c="lime",  zorder=5, label=f"Tip ({len(tip)} nodes)")

        # Use auto aspect — keeps slender beams (e.g. 200µm×5µm) legible
        ax.set_aspect("auto")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#333", edgecolor="#555", labelcolor="#ccc")
        ax.set_title(
            f"Mesh preview  —  {len(coords):,} nodes  /  {len(tris):,} elements",
            color=_FG)
        ax.set_xlabel("x (m)", color=_FG)
        ax.set_ylabel("y (m)", color=_FG)
        ax.grid(True, alpha=0.12)
        self.plot.fig.tight_layout()
        self.plot.canvas.draw()


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────

class ResultsPage(QWidget):
    def __init__(self):
        super().__init__()
        self._data  = None
        self._anim  = None

        layout = QVBoxLayout(self)

        title = QLabel("Results")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color:{_ACCENT};")
        layout.addWidget(title)

        # Summary text
        self.summary = QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setFont(QFont("Consolas", 10))
        self.summary.setMaximumHeight(155)
        layout.addWidget(self.summary)

        # ── Action row ────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.btn_save = _btn("Save Plot", "#2196F3")
        self.btn_back = _btn("New Analysis", "#607D8B")
        btn_row.addWidget(self.btn_save)
        btn_row.addWidget(self.btn_back)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # ── Animation controls ────────────────────────────────────────────
        anim_grp = QGroupBox("Deformation Animation")
        af = QHBoxLayout(anim_grp)

        af.addWidget(QLabel("Scale:"))
        self.combo_scale = QComboBox()
        self.combo_scale.addItems([
            "Auto (10% of L)", "True scale (1x)",
            "10x", "50x", "100x", "500x", "1000x", "Custom",
        ])
        self.combo_scale.setCurrentIndex(0)
        af.addWidget(self.combo_scale)

        self.sp_custom = QDoubleSpinBox()
        self.sp_custom.setRange(1, 1_000_000)
        self.sp_custom.setValue(100)
        self.sp_custom.setSuffix("x")
        self.sp_custom.setVisible(False)
        af.addWidget(self.sp_custom)

        self.btn_animate = _btn("▶  Animate", "#9C27B0", "#7B1FA2")
        self.btn_stop    = _btn("■  Stop",    "#607D8B", "#455A64")
        self.btn_static  = _btn("Static view", "#1565C0", "#0D47A1")
        self.btn_stop.setEnabled(False)
        af.addWidget(self.btn_animate)
        af.addWidget(self.btn_stop)
        af.addWidget(self.btn_static)
        af.addStretch()

        layout.addWidget(anim_grp)

        self.plot = MplPanel(13, 5.0)
        layout.addWidget(self.plot)

        # Connections
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed)
        self.btn_animate.clicked.connect(self.start_animation)
        self.btn_stop.clicked.connect(self.stop_animation)
        self.btn_static.clicked.connect(self._show_static)

    # ── helpers ───────────────────────────────────────────────────────────

    def _on_scale_changed(self, text):
        self.sp_custom.setVisible(text == "Custom")

    def _resolve_scale(self):
        """Return numeric scale factor."""
        text = self.combo_scale.currentText()
        if text == "Auto (10% of L)":
            data = self._data
            if data is None:
                return 1.0
            ux, uy = data["u"][0::2], data["u"][1::2]
            max_disp = float(np.sqrt(ux**2 + uy**2).max())
            char_L   = max(data["params"]["L"], data["params"]["W"])
            return (0.1 * char_L / max_disp) if max_disp > 1e-30 else 1.0
        if text == "True scale (1x)":
            return 1.0
        if text == "Custom":
            return float(self.sp_custom.value())
        return float(text.replace("x", ""))

    # ── static view ───────────────────────────────────────────────────────

    def show(self, data):
        self._data = data
        txt = result_summary(data["beam_type"], data["params"], data["results"])
        self.summary.setPlainText(txt)
        self._show_static()

    def _show_static(self):
        self.stop_animation()
        if self._data is None:
            return
        data = self._data
        fig = self.plot.fig
        fig.clear()
        scale = self._resolve_scale()
        plot_beam_results(
            data["coords"], data["tris"], data["u"], data["svm"],
            data["beam_type"], data["params"], data["results"],
            scale_factor=scale, fig=fig
        )
        self.plot.canvas.draw()

    # ── animation ─────────────────────────────────────────────────────────

    def start_animation(self):
        if self._data is None:
            return
        self.stop_animation()

        import matplotlib.tri as mtri

        data   = self._data
        coords = data["coords"]
        tris   = data["tris"]
        u      = data["u"]
        ux, uy = u[0::2], u[1::2]
        mag_max_raw = float(np.sqrt(ux**2 + uy**2).max())
        scale  = self._resolve_scale()

        # Pre-compute all frame data (coords + magnitude arrays)
        N_FRAMES = 60
        half = N_FRAMES // 2
        t_arr = np.concatenate([
            np.linspace(0.0, 1.0, half, endpoint=False),
            np.linspace(1.0, 0.0, N_FRAMES - half, endpoint=False),
        ])
        self._anim_frames = []
        for t in t_arr:
            s = scale * t
            cd = coords.copy()
            cd[:, 0] += s * ux
            cd[:, 1] += s * uy
            mag = np.sqrt((s * ux)**2 + (s * uy)**2) * 1e6
            self._anim_frames.append((cd, mag, s))

        self._anim_tris     = tris
        self._anim_vmax     = max(mag_max_raw * scale * 1e6, 1e-30)
        self._anim_frame_i  = 0

        # Prepare figure
        fig = self.plot.fig
        fig.clear()
        self._anim_ax = fig.add_subplot(111)
        self._anim_ax.set_facecolor(_DARK_AX)
        self._anim_ax.set_aspect("equal")

        # Colorbar: use a standalone ScalarMappable so range is always correct
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=self._anim_vmax)
        sm = ScalarMappable(norm=norm, cmap="plasma")
        sm.set_array([])
        cb = fig.colorbar(sm, ax=self._anim_ax, pad=0.02)
        cb.set_label("|u| (um)", color="#ccc")
        cb.ax.yaxis.set_tick_params(color="#ccc")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="#ccc")
        self._anim_cb = cb   # kept for reference, no update needed

        # Draw frame at peak (halfway) so first visible frame is meaningful
        cd0, mag0, s0 = self._anim_frames[len(self._anim_frames) // 4]
        tri0 = mtri.Triangulation(cd0[:, 0], cd0[:, 1], tris)
        self._anim_ax.tricontourf(tri0, mag0, levels=40, cmap="plasma",
                                   vmin=0, vmax=self._anim_vmax)
        self._anim_ax.set_xlabel("x (m)", color=_FG)
        self._anim_ax.set_ylabel("y (m)", color=_FG)
        self._anim_ax.tick_params(colors=_FG)
        for sp in self._anim_ax.spines.values():
            sp.set_color("#555")
        self._anim_ax.set_title(
            f"Deformation  —  0.0x  |  scale = {scale:.0f}x",
            color="#ddd", fontsize=10)
        self.plot.canvas.draw()

        # QTimer drives the animation (reliable in PyQt6)
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(50)   # 20 FPS
        self._anim_timer.timeout.connect(self._anim_tick)
        self._anim_timer.start()

        self.btn_animate.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _anim_tick(self):
        import matplotlib.tri as mtri
        idx = self._anim_frame_i
        cd, mag, s = self._anim_frames[idx]

        ax = self._anim_ax
        # Remove previous contour collections
        for coll in ax.collections:
            coll.remove()

        tri_f = mtri.Triangulation(cd[:, 0], cd[:, 1], self._anim_tris)
        ax.tricontourf(tri_f, mag, levels=40, cmap="plasma",
                       vmin=0, vmax=self._anim_vmax)
        ax.set_title(
            f"Deformation  —  {s:.1f}x  |  max = {mag.max():.4f} um",
            color="#ddd", fontsize=10)

        self.plot.canvas.draw_idle()
        self._anim_frame_i = (idx + 1) % len(self._anim_frames)

    def stop_animation(self):
        if hasattr(self, "_anim_timer") and self._anim_timer is not None:
            self._anim_timer.stop()
            self._anim_timer = None
        self.btn_animate.setEnabled(True)
        self.btn_stop.setEnabled(False)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────

class SolidWindow(QMainWindow):
    """Stand-alone solid mechanics wizard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MDA Solver — Solid Mechanics / Beam Analysis")
        self.setMinimumSize(900, 680)
        self.resize(1080, 760)
        self._result_data = None
        self._worker = None

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(4)
        central.setStyleSheet(_BASE_STYLE)

        # Header
        hdr = QLabel("Solid Mechanics — Beam Analysis")
        hdr.setFont(QFont("Segoe UI", 17, QFont.Weight.Bold))
        hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hdr.setStyleSheet(f"color:{_ACCENT}; padding:8px;")
        root.addWidget(hdr)

        sub = QLabel("Plane-stress FEM · Linear elasticity · Analytical comparison")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet("color:#888; font-size:10pt;")
        root.addWidget(sub)

        # Step indicator
        self.steps = StepIndicator()
        root.addWidget(self.steps)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color:#555;")
        root.addWidget(line)

        # Pages
        self.stack = QStackedWidget()
        self.page_type   = BeamTypePage()
        self.page_params = ParametersPage()
        self.page_solve  = MeshSolvePage()
        self.page_results = ResultsPage()

        for pg in (self.page_type, self.page_params,
                   self.page_solve, self.page_results):
            self.stack.addWidget(pg)
        root.addWidget(self.stack)

        # Navigation
        nav = QHBoxLayout()
        self.btn_back = _btn("< Back", "#607D8B")
        self.btn_next = _btn("Next >", "#1565C0", "#0D47A1")
        nav.addWidget(self.btn_back)
        nav.addStretch()
        nav.addWidget(self.btn_next)
        root.addLayout(nav)

        # Connections
        self.btn_back.clicked.connect(self._go_back)
        self.btn_next.clicked.connect(self._go_next)
        self.page_solve.btn_run.clicked.connect(self._run_analysis)
        self.page_results.btn_save.clicked.connect(self._save_plots)
        self.page_results.btn_back.clicked.connect(self._restart)

        self._go_to(0)

    # ── navigation ────────────────────────────────────────────────────────────

    def _go_to(self, idx):
        self.stack.setCurrentIndex(idx)
        self.steps.set_active(idx)
        self.btn_back.setEnabled(idx > 0)
        self.btn_next.setVisible(idx < 3)

        if idx == 1:
            # Show/hide zig-zag fields
            zz = (self.page_type.beam_type == "zigzag")
            self.page_params.show_zigzag_fields(zz)

        if idx == 2:
            self.btn_next.setEnabled(False)   # enabled after solve completes

        if idx == 3:
            self.btn_next.setVisible(False)

    def _go_back(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self._go_to(idx - 1)

    def _go_next(self):
        idx = self.stack.currentIndex()
        if idx < 3:
            self._go_to(idx + 1)

    # ── analysis ──────────────────────────────────────────────────────────────

    def _run_analysis(self):
        p = self.page_params.get_params()
        beam_type = self.page_type.beam_type

        self.page_solve.btn_run.setEnabled(False)
        self.page_solve.progress.setVisible(True)
        self.page_solve.console.clear_log()
        self.btn_next.setEnabled(False)
        self._result_data = None

        self._worker = BeamWorker(beam_type, p)
        self._worker.log_signal.connect(self.page_solve.console.log)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, data):
        self._result_data = data
        self.page_solve.progress.setVisible(False)
        self.page_solve.btn_run.setEnabled(True)

        # Show mesh preview
        self.page_solve.show_mesh(
            data["coords"], data["tris"], data["fixed"], data["tip"])

        self.btn_next.setEnabled(True)
        self.page_solve.console.log("Analysis complete — click Next to view results.")

        # Pre-populate results page
        self.page_results.show(data)

    def _on_error(self, msg):
        self.page_solve.progress.setVisible(False)
        self.page_solve.btn_run.setEnabled(True)
        self.page_solve.console.log(f"ERROR:\n{msg}")
        QMessageBox.critical(self, "Analysis Error", msg)

    # ── save plots ────────────────────────────────────────────────────────────

    def _save_plots(self):
        if self._result_data is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Results Plot", "beam_results.png",
            "PNG Images (*.png);;PDF Files (*.pdf)")
        if path:
            self.page_results.plot.fig.savefig(
                path, dpi=180, bbox_inches="tight", facecolor=_DARK_BG)
            self.page_results.summary.appendPlainText(f"\nPlot saved → {path}")

    def _restart(self):
        self._result_data = None
        self._go_to(0)
