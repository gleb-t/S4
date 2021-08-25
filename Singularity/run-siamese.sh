#!/bin/bash

set -o errexit

echo "DEBUG: Args received: $*"
# Take the last two args.
config=${*: -2:1}
runsetId=${*: -1}

taskId=${SLURM_ARRAY_TASK_ID:-0}  # Take the env var or use zero as the default.
taskNumber=${SLURM_ARRAY_TASK_COUNT:-1}  # Take the env var or use zero as the default.

echo "Starting the S4 run with config '$config'"
python ./src/Siamese/siamese.py --config-path /app/config/$config.py --runset-id $runsetId \
                                --task-id $taskId --task-number $taskNumber

