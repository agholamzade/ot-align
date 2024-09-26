#!/bin/bash -l


#SBATCH -D ./

#SBATCH --array=1-16

#SBATCH -o ./jobs_log/job_%A_%a.out        
#SBATCH -e ./jobs_log/job_%A_%a.err       

#SBATCH -J fetures_morph
#
#SBATCH --ntasks=1
#SBATCH --constraint="gpu"

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=125000


#SBATCH --mail-type=none
#SBATCH --mail-user=ali.gholamzadeh@tuebingen.mpg.de
#SBATCH --time=18:00:00


# Default mode
mode="morph_discrete"

# Parse command-line arguments for mode
while getopts m: flag
do
    case "${flag}" in
        m) mode=${OPTARG};;
    esac
done


module purge 
module load intel/21.2.0 impi/2021.2 cuda/11.2
module load anaconda/3/2023.03

source venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python main.py --config=./configs/features/features.py  --exp=feature_morph --mode=$mode