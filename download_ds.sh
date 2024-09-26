#!/bin/bash -l


#SBATCH -D ./
#SBATCH --array=0-8

#SBATCH -o ./jobs_log/job_%A_%a.out
#SBATCH -e ./jobs_log/job_%A_%a.err


#SBATCH -J download_ds


#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72   

#SBATCH --mem=120000

#SBATCH --time=24:00:00

ARGS=("train[0%:10%]" "train[10%:20%]" "train[20%:30%]" "train[30%:40%]" "train[50%:60%]" "train[60%:70%]" "train[70%:80%]" "train[80%:90%]" "train[90%:100%]")

module purge 
module load intel/21.4.0 impi/2021.4
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source venv/bin/activate

srun python download_ds.py ${ARGS[$SLURM_ARRAY_TASK_ID]}
