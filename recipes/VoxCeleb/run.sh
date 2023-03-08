#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: run.sh
# time: 2022/5/18 19:32
# Description: 

stage=0

waited=0
while [ $(ps 217457 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then

  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port='29544' SpeakerRec/train_speaker_embeddings.py SpeakerRec/hparams/train_ecapa_tdnn.yaml --distributed_launch --distributed_backend='nccl'
#  --auto_mix_prec

#  CUDA_VISIBLE_DEVICES=5 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_plda.py SpeakerRec/hparams/verification_plda_ecapa.yaml

  exit
fi