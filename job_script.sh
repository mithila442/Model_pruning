#!/bin/bash
#SBATCH -A ASC23013                              # Account name
#SBATCH -J DDP_job                               # Job name
#SBATCH -o DDP_job.o%j.out                           # Name of stdout output file (%j expands to jobId)
#SBATCH -p gpu-a100-dev                          # Queue (partition) name
#SBATCH -N 1                                     # Total number of nodes requested
#SBATCH -n 1                                     # Total number of mpi tasks requested
#SBATCH -t 02:00:00                              # Run time (hh:mm:ss)

# Run the pruning script using PyTorch's distributed launcher
# --nproc_per_node=3 [This is based on number of GPUs available on the system]
python finetune1.py --prune

