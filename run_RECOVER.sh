#!/usr/bin/env bash
DATA=cifar10
BATCHSIZE=(128)
EPOCH=120
MODEL=resnet20
CHECKPOINTS=ckps
R=.pth.tar
LR=(0.5 0.1)
AT=(0.8 0.8)
INITW=(2 3 4 5 6)
INIT_LAMDA=(100)
LAMDA=(1 5 10 20 100)
IMRATIO=(0.02)
DR=(2)
GPUS=(3,5 2,3)

for ((at=0; at<1; at++)); do
{
for((ibd=0; ibd<1; ibd++)); do
{
for((im=0; im<1; im++)); do
{
for((llr=0; llr<1; llr++)); do
{
for((in=0; in<1; in++)); do
{
for((lbd = 0; lbd<1; lbd++)); do
{
     ALGNAME=RECOVER
        python3 -W ignore RECOVER.py \
            --dataset $DATA \
            --model $MODEL \
            --save ${DATA}/${ALGNAME}/${ALG}_${ALGNAME}_lr_${LR[$llr]}_at_${AT[$llr]}_IMRATIO_${IMRATIO[$im]}_lambda_${INIT_LAMDA[$ilbd]}_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_10_STAGE_4_on_time_init_${INITW[$in]} \
            --res_name ${ALG}_${ALGNAME}_lr_${LR[$llr]}_at_${AT[$llr]}_IMRATIO_${IMRATIO[$im]}_lambda_${INIT_LAMDA[$ilbd]}_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_10_STAGE_4_on_time_init_${INITW[$in]} \
            --epochs ${EPOCH} \
            --batch-size ${BATCHSIZE} \
            --lr ${LR[$llr]} \
            --a_t ${AT[$llr]} \
            --ith_init_run ${INITW[$in]} \
            --restart_init_loop 1 \
            --lamda ${LAMDA[$lbd]} \
            --epochs ${EPOCH} \
            --print_freq 50 \
            --im_ratio ${IMRATIO[$im]} \
            --init_lamda ${INIT_LAMDA[$ilbd]} \
            --alg ${ALGNAME} \
            --gpus ${GPUS[$((im % 2))]} \
            --resume False \
            --alg ${ALGNAME}
}
done
}
done
}
done
}
done
}
done
}
done



# gpus ${GPUS[$((im % 2))]} \
#         echo ${GPUS[$((im % 2))]}
