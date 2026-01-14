# Fusion
Helper files for Fusion stuffs

DEPRECATED: fullwavedensfile.py and fullwavemagfile.py transpose our ne and B data for ERMES (original ones made by Matthew Liang)
scotty2ERMES.py generates the points and parameters needed for ERMES and has functions for plotting to compare Scotty and ERMES results.

gen2Dfullwavefile.py and gen3Dfullwavefile.py are consolidated and tidied up.

createAB.py, PETSc_batch.sh, createVectorX.py are for running PETSc external solver with GiD.

Workflow:

Run a SCOTTY simulation -> run get_ERMES_parameters() -> create your GiD/ERMES problem set, mesh, run in debug -> Run gen2D/3Dfullwavefile.py -> update plasma -> Run ERMES -> createAB.py -> PETSc_batch.sh -> createVectorX -> Run ERMES in read mode -> save results as ASCII -> run ERMES_results_to_plot

A .h5 file of compiled useful results will be saved