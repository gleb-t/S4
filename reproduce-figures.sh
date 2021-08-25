#!/bin/bash
set -e

RECOMPUTE_ARG=""
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        -r|--recompute)
            RECOMPUTE_ARG="--recompute"
            shift
            ;;
        *)        # unknown option
            echo "Unknown arg: ${1}"
            exit 1
            ;;
    esac
done


export DEV_VOLUME_DATA_PATH=./data
export DEV_OUT_PATH=./out
export DEV_PYPLANT_PATH=./pyplant
export DEV_SIAMESE_CONFIG_PATH=./configs

echo "Creating the PyPlant dir at ${DEV_PYPLANT_PATH}."
/usr/bin/mkdir -p ${DEV_PYPLANT_PATH}
echo "Creating the output dir at ${DEV_OUT_PATH}"
/usr/bin/mkdir -p ${DEV_OUT_PATH}

CONTAINER_ARGS="" 
CONTAINER_ARGS+="--nv " 
CONTAINER_ARGS+="--no-home " 
CONTAINER_ARGS+="--cleanenv " 
CONTAINER_ARGS+="--pwd /app/src " 
CONTAINER_ARGS+="--bind ${DEV_OUT_PATH}:/app/out " 
CONTAINER_ARGS+="--bind ${DEV_PYPLANT_PATH}:/app/plant " 
CONTAINER_ARGS+="--bind ${DEV_SIAMESE_CONFIG_PATH}:/app/config " 
CONTAINER_ARGS+="--bind ${DEV_VOLUME_DATA_PATH}:/app/data "

if [[ -n "${RECOMPUTE_ARG}" ]]; then
    echo "Recomputing the full results. This will take ~50h due to the EMD baselines!"
    sleep 5
    /usr/bin/singularity exec \
        $CONTAINER_ARGS \
        s4-image.sif \
        python ./Siamese/siamese.py \
            --config-path /app/config/200817_cylinder-300-2-basic_all-metrics.py \
            --runset-id recomputed-runset \
            --run-id recomputed-run
fi

# Fixes the issue: https://github.com/sylabs/singularity/issues/2203
echo "Making sure that the NVIDIA kernel module is loaded."
/bin/bash -c '/usr/bin/nvidia-modprobe -u -c=0'

echo "Generating the table CSV."
/usr/bin/singularity exec \
    $CONTAINER_ARGS \
    s4-image.sif \
    python ./Siamese/scripts/export_paper_csv.py "${RECOMPUTE_ARG}"


echo "Generating the matches figure."
/usr/bin/singularity exec \
    $CONTAINER_ARGS \
    s4-image.sif \
    python ./Siamese/scripts/render_matches_figure.py "${RECOMPUTE_ARG}"


#"/bin/bash -c 'cd /app/src'" # && python ./Siamese/scripts/export_paper_csv.py"
#
#
#"cd /app/src/"
#s4-image.sif "cd /app/src/ && python ./Siamese/scripts/export_paper_csv.py"

