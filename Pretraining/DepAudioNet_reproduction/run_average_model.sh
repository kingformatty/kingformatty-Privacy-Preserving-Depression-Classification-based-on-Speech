#! /usr/bin/env bash
root_dir=/home/jinhan/Dataset/librispeech/LibriSpeech360/audio_feats/feats_DepAudioNet/mel_svn_exp/exp_BN_20/
exp_dir=TM_tmp10_distinct
num=10
end_epoch=89
start_epoch=`echo ${end_epoch}-${num} | bc`
out_name=average_${start_epoch}_${end_epoch}.pth
echo $out_name

model_dir=${root_dir}/${exp_dir}/

exp_run/average_checkpoint.py \
    --exp_dir $model_dir \
    --out_name $out_name \
    --last_epoch $end_epoch \
    --num $num

echo "Average checkpoints Finished."
