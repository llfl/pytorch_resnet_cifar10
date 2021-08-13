#!/bin/bash
t="`date +%y%m%d%H%M%S`"
for model in rpcnet
do
    echo "python -u trainer.py -e --arch=$model"
    python -u trainer.py --quan -e --arch=$model --load=save_rpcnet210811091330 |& tee -a log_eval_$model$t
done
