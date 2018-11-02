#!/bin/bash
# Usage Main-wus-soft-train-pos.sh model_file_name data_folder data_prefix use_aux_loss_and/or_pos_feat
# ./Main-wus-soft-train-pos.sh norm_soft wus/phase2/btagger wus
# ./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger wus
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus pos
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus aux
# ./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus pos_aux
##########################################################################################


export TRAIN=$2/train_silverpos.txt
export DEV=$2/dev_autopos.txt
export TEST=$2/test_autopos.txt

export PR="$3_phase2"
export MODEL=$1
if [[ $4 == "aux" ]]; then
export PR="${PR}/$3_aux"
elif [[ $4 == "pos" ]]; then
export PR="${PR}/$3_pos"
elif [[ $4 == "pos_aux" ]]; then
export PR="${PR}/$3_pos_aux"
else
export PR=${PR}/$3
fi
echo "$PR"


########### train + eval of individual models
for (( k=1; k<=5; k++ ))
do
(
if [[ $4 == "aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40  --aux_pos_task
elif [[ $4 == "pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40  --pos_feature
elif [[ $4 == "pos_aux" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40  --pos_feature --aux_pos_task
elif [[ $1 == "norm_soft_pos" ]]; then
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40
else
PYTHONIOENCODING=utf8 python ${MODEL}.py train --dynet-seed $k --train_path=$TRAIN --dev_path=$DEV ${PR}_${MODEL}_$k  --epochs=40
fi

wait

PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$DEV --beam=3 --pred_path=best.dev.3   &
PYTHONIOENCODING=utf8 python ${MODEL}.py test ${PR}_${MODEL}_$k --test_path=$TEST --beam=3 --pred_path=best.test.3
) &
done

wait

########### Evaluate ensemble 5

PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$DEV --beam=3 --pred_path=best.dev.3 ${PR}_${MODEL}_ens5   &
PYTHONIOENCODING=utf8 python ${MODEL}.py ensemble_test ${PR}_${MODEL}_1,${PR}_${MODEL}_2,${PR}_${MODEL}_3,${PR}_${MODEL}_4,${PR}_${MODEL}_5 --test_path=$TEST --beam=3 --pred_path=best.test.3 ${PR}_${MODEL}_ens5  

