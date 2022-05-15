#!/usr/bin/env bash

# author: yangwenhao
# contact: 874681044@qq.com
# file: run.sh
# time: 2022/3/20 01:05
# Description: 

stage=0

waited=0
while [ $(ps 217457 | wc -l) -eq 2 ]; do
  sleep 60
  waited=$(expr $waited + 1)
  echo -en "\033[1;4;31m Having waited for ${waited} minutes!\033[0m\r"
done

if [ $stage -le 0 ]; then

#  CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port='29501' train_speaker_embeddings.py SpeakerRec/hparams/train_ecapa_tdnn.yaml --distributed_launch --distributed_backend='nccl'

  CUDA_VISIBLE_DEVICES=5 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_plda.py SpeakerRec/hparams/verification_plda_ecapa.yaml

  exit
fi

if [ $stage -le 1 ]; then

  CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch --nproc_per_node=2 train_speaker_embeddings.py SpeakerRec/hparams/train_x_vectors.yaml --distributed_launch --distributed_backend='nccl'

fi


if [ $stage -le 10 ]; then

#  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_speaker_embeddings_end2end.py SpeakerRec/hparams/train_sinc_ecapa_tdnn.yaml --distributed_launch --distributed_backend='nccl'


  CUDA_VISIBLE_DEVICES=0 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_plda.py SpeakerRec/hparams/verification_plda_ecapa.yaml

  exit
fi


if [ $stage -le 20 ]; then

  CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train_speaker_embeddings.py SpeakerRec/hparams/train_agg_tdnn.yaml --distributed_launch --distributed_backend='nccl'


#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_cosine.py SpeakerRec/hparams/verification_ecapa.yaml
#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_plda.py SpeakerRec/hparams/verification_plda_ecapa.yaml

  exit
fi

if [ $stage -le 30 ]; then

  CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --master_port='29505' --nproc_per_node=2 train_speaker_embeddings.py SpeakerRec/hparams/train_resnet.yaml --distributed_launch --distributed_backend='nccl'


#  CUDA_VISIBLE_DEVICES=3 python speaker_verification_cosine.py SpeakerRec/hparams/verification_res2net.yaml
#  CUDA_VISIBLE_DEVICES=0 python speaker_verification_plda.py SpeakerRec/hparams/verification_plda_ecapa.yaml


  exit
fi