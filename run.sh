BASE_DIR_LOCAL=/home/kloping/OpenSet_MOT/
BASE_DIR_TUM=/storage/slurm/liuyang/

python sort.py --seq_path /storage/slurm/liuyang/Tracking/proposals/forSORT/ \
               --outdir /storage/slurm/liuyang/Trackng/SORT_results/ \
               --phase val/ --min_hits 0

python det_conversion --root_dir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/boxNMS/_objectness/ \
                      --outdir /storage/slurm/liuyang/TAO_eval/TAO_VAL_Proposals/Panoptic_Cas_R101_NMSoff_forTracking/forSORT_masks/ \
                      --scoring objectness
