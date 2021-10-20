#!/bin/bash
#SBATCH --array=1-20
#SBATCH --account=def-lplevass
echo $SLURM_ARRAY_TASK_ID
