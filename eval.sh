#!/bin/bash
t="`date +%y%m%d%H%M%S`"
for model in rpcnet_deploy
do
    echo "python -u trainer.py -e --arch=$model"
    python -u trainer.py -e --arch=$model
done
