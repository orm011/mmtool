#! /bin/bash

set -x
source ~/.bashrc 
sync_conda_global_to_worker
source ~/.bashrc
export __MODIN_AUTOIMPORT_PANDAS__=1
mamba activate mmtool

BASE_DIR=$(dirname `python -c 'import mmtool; print(mmtool.__path__[0])'`)
DIR="$BASE_DIR/scripts/"

SIGFILE="$HOME/ray2.head"
# change current ray head
echo '' > $SIGFILE

bash +x $DIR/start_worker.bash --head 

python -c 'import ray; ray.init("auto", namespace="mmtool"); print(ray.available_resources());'
sleep infinity