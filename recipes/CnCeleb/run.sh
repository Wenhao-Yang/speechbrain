#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: run.sh
# time: 2022/3/20 01:05
# Description: 

stage=10

waited=0
while [ $(ps 117765 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then

  CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=2 train_speaker_embeddings.py SpeakerRec/hparams/train_ecapa_tdnn.yaml --distributed_launch --distributed_backend='nccl'

#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
  exit
fi

if [ $stage -le 5 ]; then

#  CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 train_speaker_embeddings.py SpeakerRec/hparams/train_ecapa_tdnn.yaml --distributed_launch --distributed_backend='nccl'

  CUDA_VISIBLE_DEVICES=0 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
  exit
fi


if [ $stage -le 10 ]; then

  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_speaker_embeddings.py SpeakerRec/hparams/train_x_vectors.yaml --distributed_launch --distributed_backend='ddp_nccl'

fi