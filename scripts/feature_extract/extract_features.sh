#!/bin/bash -l


#SBATCH -D ./

#SBATCH -o ./jobs_log/job.out.%j
#SBATCH -e ./jobs_log/job.err.%j


#SBATCH -J feature_extract
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --partition=gpudev
# --- uncomment to use 2 GPUs on a shared node ---
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=36
#SBATCH --mem=250000

# --- uncomment to use 4 GPUs on a full node ---
# #SBATCH --gres=gpu:a100:4
# #SBATCH --cpus-per-task=72
# #SBATCH --mem=500000
#
#SBATCH --mail-type=none
#SBATCH --mail-user=ali.gholamzadeh@tuebingen.mpg.de
#SBATCH --time=00:15:00


module purge 
module load intel/21.2.0 impi/2021.2 cuda/12.1
module load anaconda/3/2023.03
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

source hf_venv/bin/activate


srun python src/feature_extraction/extract.py