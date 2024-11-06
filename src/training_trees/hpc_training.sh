#!/bin/bash

#SBATCH --job-name=train_svl_house_1

#SBATCH --time=2-12:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=20

#SBATCH --ntasks-per-node=1

#SBATCH --partition=skylake,skylake_mpi

#SBATCH --mem-per-cpu=800M

#SBATCH --array=0-4

cd /user/brussel/102/vsc10250/interconnect/smart-houses-ems

module load Python/3.10.4-GCCcore-11.3.0
module load Boost/1.79.0-GCC-11.3.0
module load tbb/2021.5.0-GCCcore-11.3.0
module load Eigen/3.4.0-GCCcore-11.3.0
module load NLopt/2.7.1-GCCcore-11.3.0
module load matplotlib/3.5.2-foss-2022a

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/user/brussel/102/vsc10250/.local/lib64

source $VSC_DATA/svl_ems/bin/activate
python training_trees/hpc_run.py 1