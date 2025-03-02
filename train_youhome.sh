PYTHON="/home/aperture/miniconda3/envs/torch/bin/python"

DATE=`date +%Y-%m-%d`

model=resnet
epochs=100
model_depth=152
batch_size=64
n_classes=31

dataset=youhome

root_path="/media/aperture/Delta/Descartes/mp4data_2021_frames"
video_path="img"
annotation_path="anno/YouHome.json"
result_path="results"
# pretrain_path="pretrain/mobilenet.pth"
# pretrain_path="pretrain/mobilenetv2.pth"
# pretrain_path="pretrain/shufflenet.pth"
# pretrain_path="pretrain/shufflenetv2.pth"
# pretrain_path="pretrain/squeezenet.pth"
# pretrain_path="pretrain/r3d18_KM_200ep.pth"
# pretrain_path="pretrain/r3d50_KMS_200ep.pth"
# pretrain_path="pretrain/r3d34_KM_200ep.pth"
# pretrain_path="pretrain/r3d101_KM_200ep.pth"
# pretrain_path="pretrain/r3d152_KM_200ep.pth"
pretrain_path="results/2022-12-14/youhome_resnet152_200_64_0/save_200.pth"
# pretrain_path="results/2022-12-14/youhome_shufflenet_200_16_0/save_200.pth"
# pretrain_path="results/2022-12-15/youhome_squeezenet_100_32_0/save_100.pth"
# pretrain_path="results/2022-12-14/youhome_shufflenetv2_200_16_0/save_200.pth"

SUFFIX=0

if [ ! -d "$DIRECTORY" ]; then
    mkdir ${root_path}/${result_path}/${DATE}
fi

mkdir ${root_path}/${result_path}/${DATE}/${dataset}_${model}${model_depth}_${epochs}_${batch_size}_${SUFFIX}

cp train_youhome.sh ${root_path}/${result_path}/${DATE}/${dataset}_${model}${model_depth}_${epochs}_${batch_size}_${SUFFIX}

$PYTHON main.py \
    --root_path ${root_path} \
    --video_path ${video_path} \
    --annotation_path ${annotation_path} \
    --result_path ${result_path}/${DATE}/${dataset}_${model}${model_depth}_${epochs}_${batch_size}_${SUFFIX} \
    --dataset ${dataset} \
    --n_classes ${n_classes} \
    --learning_rate 0.001 \
    --model ${model} \
    --batch_size ${batch_size} \
    --n_epochs ${epochs} \
    --n_threads 16 \
    --checkpoint 20 \
    --n_pretrain_classes 31 \
    --multistep_milestones 25 50 75\
    --pretrain_path ${pretrain_path} \
    --model_depth ${model_depth} \
