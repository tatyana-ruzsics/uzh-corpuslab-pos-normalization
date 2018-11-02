

###### Context-free

# train nmt
#./Main-wus-soft-train-pos.sh norm_soft wus/phase2/btagger wus
# nmt detailed eval
./Main-wus-sync-pos.sh btagger wus wus_phase2 5 3 nmt norm_soft
# sync decoding
./Main-wus-sync-pos.sh btagger wus wus_phase2 5 3 we norm_soft


###### Context-aware

# train nmt
#./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/treetagger wus_tt
#./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger wus_bt
#./Main-wus-soft-train-pos.sh norm_soft_pos wus/phase2/btagger-sms wus_bt_sms

# nmt detailed eval
./Main-wus-sync-pos.sh btagger wus_bt wus_phase2 5 3 nmt norm_soft_pos
#./Main-wus-sync-pos.sh btagger-sms wus_bt_sms wus_phase2 5 3 nmt norm_soft_pos
#./Main-wus-sync-pos.sh treetagger wus_tt wus_phase2 5 3 nmt norm_soft_pos

# sync decoding
./Main-wus-sync-pos.sh btagger wus_bt wus_phase2 5 3 we norm_soft_pos
#./Main-wus-sync-pos.sh btagger-sms wus_bt_sms wus_phase2 5 3 we norm_soft_pos
#./Main-wus-sync-pos.sh treetagger wus_tt wus_phase2 5 3 we norm_soft_pos

###### Extra analysis
# Baseline
python accuracy-det.py eval_baseline ../data/wus/phase2/btagger/train_silverpos.txt ../data/wus/phase2/btagger/test_autopos.txt --error_file=baseline_wus_errors.txt
python accuracy-det.py eval_ambiguity_baseline ../data/wus/phase2/btagger/train_silverpos.txt ../data/wus/phase2/btagger/test_autopos.txt --input_format=0,1,2 --error_file=baseline_pos_wus_errors.txt


##### Extra
#
#./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/treetagger wus_tt
#./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/treetagger wus_tt aux
#./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/treetagger wus_tt
#./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/treetagger wus_tt aux
#./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger-sms wus_bt_sms aux
#./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/btagger-sms wus_bt_sms aux
#
#./Main-wus-soft-train-pos.sh norm_soft_context wus/phase2/btagger wus_bt aux
#./Main-wus-soft-train-pos.sh norm_soft_char_context wus/phase2/btagger wus_bt aux
