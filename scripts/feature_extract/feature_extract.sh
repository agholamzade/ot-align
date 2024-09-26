#!/bin/bash -l


#SBATCH -D ./

#SBATCH -o ./jobs_log/job.out.%j
#SBATCH -e ./jobs_log/job.err.%j


#SBATCH -J feature_extract
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=250000

#SBATCH --mail-type=none
#SBATCH --mail-user=ali.gholamzadeh@tuebingen.mpg.de
#SBATCH --time=05:00:00


module purge 
mode load cuda/12.1
module load anaconda/3/2023.03

source hf_venv/bin/activate


srun python src/large_models/feature_extraction.py