#!/usr/bin/env bash
DATA=cifar10
BATCHSIZE=(512)
EPOCH=120
MODEL=resnet20
CHECKPOINTS=ckps
R=.pth.tar
LR=(0.5)
INITW=(2)
LAMDA=(1 2 3 4)
INIT_LAMDA=(30)
IMRATIO=(0.02 0.05 0.1 0.2)
GPUS=(0 1 2 3)



for((lbd=0; lbd<4; lbd++)); do
{
for((llr=0; llr<1; llr++)); do
{
for((in=0; in<1; in++)); do
{
for((im=3;im<4;im++));do
{
for((ilbd = 0; ilbd<1; ilbd++)); do
{
        ALGNAME=MBDRO
         python3 -W ignore MBDRO.py\
            --dataset $DATA \
            --model $MODEL \
            --save ${DATA}/${ALGNAME}/${ALGNAME}_lr_${LR[$llr]}_IMRATIO_${IMRATIO[$im]}_lambda_${INIT_LAMDA[$ilbd]}_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_init_${INITW[$in]}_3_lambda\
            --res_name ${ALGNAME}_lr_${LR[$llr]}_IMRATIO_${IMRATIO[$im]}_lambda_${INIT_LAMDA[$ilbd]}_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_init_${INITW[$in]}_3_lambda\
            --epochs ${EPOCH} \
            --batch-size ${BATCHSIZE} \
            --lr ${LR[$llr]} \
            --stages [0,60] \
            --ith_init_run ${INITW[$in]} \
            --restart_init_loop 1 \
            --lamda ${LAMDA[$lbd]} \
            --init_lamda ${INIT_LAMDA[$ilbd]} \
            --epochs ${EPOCH} \
            --print_freq 50 \
            --momentum 0.9 \
            --im_ratio ${IMRATIO[$im]} \
            --gpus ${GPUS[$lbd]}
}
done
}&
done
}
done
}
done
}
done
# Some thing wrong with the code.
# Implement the momentum based code.

