# Runs PETSc Solver
# Taken from various PETSc and ERMES scripts.
#
# Written by Dylan James Mc Kaige & Ray Ng
# Created on 14/5/2025
# Updated on 19/5/2025

# Path to GiD, need to manually change
MY_PATH=$HOME/mckaigedj/7_degree_new/7_degree_new.gid/

MPI_EXE=$PETSC_DIR/lib/petsc/bin/petscmpiexec
PETSC_EXE=$HOME/mckaigedj/PETSc/Solver/PETScSolver
SOLVER_INFO=$MY_PATH/SolverInfo.info

PETSC_SOLVER='-ksp_type gmres -pc_type lu -pc_factor_mat_solver_type mumps'

# It should only take 1 iter, but here we set 10000 as an arbitrary cap before it 'gives up'
SOLVER_PARAMETERS='-ksp_max_it 10000 -ksp_monitor_true_residual'
echo "Solving A*X = B, run createVectorX.py once done"
$MPI_EXE -n 8 $PETSC_EXE $PETSC_SOLVER $SOLVER_PARAMETERS -my_folder $MY_PATH > $SOLVER_INFO
