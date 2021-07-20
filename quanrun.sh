#!/bin/bash
t="`date +%y%m%d%H%M%S`"
for model in rpcnet
do
    echo "python -u trainer.py --quan --arch=$model --save-dir=save_quan_$model$t |& tee -a log_quan_$model$t"
    python -u trainer.py --quan --arch=$model  --save-dir=save_quan_$model$t |& tee -a log_quan_$model$t
done
