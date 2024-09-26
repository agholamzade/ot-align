#!/bin/bash -l


#SBATCH -D ./

#SBATCH -o ./jobs_log/job.out.%j
#SBATCH -e ./jobs_log/job.err.%j


#SBATCH -J swiss_roll_to_spiral
#
## SBATCH --partition=gpudev

#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000

#SBATCH --mail-type=none
#SBATCH --mail-user=ali.gholamzadeh@tuebingen.mpg.de
#SBATCH --time=10:00:00
# #SBATCH --time=00:15:00


module purge 
module load anaconda/3/2023.03

source venv/bin/activate


srun python main.py --config=./configs/swiss_roll/swiss_roll.py --mode=morph --exp=swiss_roll