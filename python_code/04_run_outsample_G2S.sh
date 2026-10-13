#!/bin/bash
#SBATCH --job-name=enso_outsample_G2S
#SBATCH --time=2-12:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-11%4
#SBATCH --cpus-per-task=16
#SBATCH --mem=4G
#SBATCH --partition=defq
##SBATCH --constraint="arch=skylake"  # uncomment if this architecture is indeed available
#SBATCH --output=logs/%x_%A_%a.out

echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"  

# Load the required Conda/Miniconda module  
module load <name of condo or miniconda module>

# Initialize Conda (replace with the path on your system if needed)
source <path_to_conda>/etc/profile.d/conda.sh 

# Activate the environment constructed using environment.yml
conda activate <your_conda_environment>

# Change to the project directory
cd <path_to_project>/python_code

# Make the project importable
export PYTHONPATH=<path_to_project>:$PYTHONPATH

# Run the script
python -u 04_run_outsample_G2S.py

