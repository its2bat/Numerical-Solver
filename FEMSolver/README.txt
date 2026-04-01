Run Instructions (FEMSolver Codes)

***Requirements
- Python 3.x
- Packages: `numpy`, `scipy`, `gmsh`, `pyvista`
- Gmsh must be installed and importable from Python (`import gmsh` must work)

***Files
- `FEMSolver_Project.py` : problem/geometry setup (no solve)
- `FEMSolver_Project_Mesh.py` : mesh generation + mesh preview
- `FEMSolver_Solver.py` : 2D steady solve
- `FEMSolver_Solver_Transient.py` : 2D transient solve
- `FEMSolver_Solver_Transient_ThermalPaste.py` : 2D transient with paste layer
- `Final_final_final_3d_transient_thermalpaste_mesh_full_sections.py` : 3D transient + mesh preview + full T preview + slicer GUI
- `Final_final_final_3d_steady_thermalpaste_mesh_full_sections.py` : 3D steady + mesh preview + full T preview + slicer GUI

***How to Run
1) Open the desired `.py` file in Spyder (or run from terminal).
2) Run the script.

***3D steady/transient scripts (PyVista window order)
1. Mesh preview window opens → **close it** to continue
2. Whole-object temperature preview opens → **close it** to continue
3. Interactive slicer opens (X/Y/Z sliders)

***Output
- The 3D scripts export a VTU file for ParaView.

***Notes
- Runtime depends on mesh size and PC performance (especially transient cases).
- If PyVista is missing, the script will still export VTU and you can open it in ParaView.
