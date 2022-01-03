#!/usr/bin/env bash
DATALIST=(cifar10 cifar100)
BATCHSIZE=(128)
EPOCH=120
MODEL=resnet32
CHECKPOINTS=ckps
R=.pth.tar
LR=(1)
AT=(0)
LAMDA=(5)
INITW=(2 3 4 5 6)
IMRATIO=(0.02 0.05 0.1 0.2)
DR=10
GPUS=(2)

for((da=0; da<1; da++)); do
{
for((im=0; im<1; im++)); do
{
for ((at=0; at<1; at++)); do
{
for((llr=0; llr<1; llr++)); do
{
for((in=0; in<1; in++)); do
{
         ALGNAME=SGD
         python3 -W ignore SGD.py \
            --dataset ${DATALIST[$da]} \
            --model $MODEL \
            --save ${DATALIST[$da]}/${ALGNAME}/${ALGNAME}_lr_${LR[$llr]}_at_${AT[$at]}_IMRATIO_${IMRATIO[$im]}_lambda_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_STAGE_2_init_${INITW[$in]} \
            --res_name ${ALGNAME}_lr_${LR[$llr]}_at_${AT[$at]}_IMRATIO_${IMRATIO[$im]}_lambda_${LAMDA[$lbd]}_batch_${BATCHSIZE}_epochs_${EPOCH}_model_${MODEL}_DR_${DR}_STAGE_2_init_${INITW[$in]} \
            --epochs ${EPOCH} \
            --batch-size ${BATCHSIZE} \
            --gpus ${GPUS[$in]} \
            --lr ${LR[$llr]} \
            --a_t ${AT[$at]} \
            --ith_init_run ${INITW[$in]} \
            --restart_init_loop 1 \
            --lamda ${LAMDA[$lbd]} \
            --epochs ${EPOCH} \
            --print_freq 50 \
            --momentum 0.9 \
            --im_ratio ${IMRATIO[$im]} \
            --DR ${DR}
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

# Some thing wrong with the code.
# Implement the momentum based code.

