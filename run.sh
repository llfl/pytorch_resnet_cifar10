#!/bin/bash
t="`date +%y%m%d%H%M%S`"
for model in rpcnet_deploy
do
    echo "python -u trainer.py  --arch=$model  --save-dir=save_$model$t |& tee -a log_$model$t"
    python -u trainer.py  --arch=$model  --save-dir=save_$model$t |& tee -a log_$model$t
done
