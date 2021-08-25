#!/bin/bash
set -e

singularity run library://sylabsed/examples/lolcow
echo "Called Singularity. You should see a cow above."
sleep 3

nvidia-smi || /usr/bin/nvidia-smi
echo "Called nvidia-smi. You should see your GPU info above."
sleep 3

echo "Pulling LFS files"
git lfs pull

echo "Unpacking the data."
(cd data; unzip -o cylinder-ensemble.zip)
(cd out; unzip -o 20200817-153710_cluster-1125_200817_cylinder-300-2-basic_all-metrics.zip)

echo "Building the image. Root required."
sudo singularity build -F s4-image.sif Singularity/s4-image.def

echo "Reproducing the figures."
chmod u+x reproduce-figures.sh
./reproduce-figures.sh

echo "Done."
