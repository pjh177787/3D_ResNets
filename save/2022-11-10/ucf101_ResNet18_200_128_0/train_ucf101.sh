PYTHON="/home/aperture/anaconda3/envs/torch/bin/python"

DATE=`date +%Y-%m-%d`

epochs=200
model_depth=18
batch_size=128

dataset=ucf101

root_path="/mnt/echo/Euler/3dcnn_data/ucf101"
video_path="jpg"
annotation_path="anno/ucf101_01.json"
result_path="results"
pretrain_path="pretrain/r3d18_KM_200ep.pth"

SUFFIX=0

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save
    mkdir ./save/${DATE}/
fi

mkdir ./save/${DATE}/${dataset}_ResNet${model_depth}_${epochs}_${batch_size}_${SUFFIX}

cp train_ucf101.sh ./save/${DATE}/${dataset}_ResNet${model_depth}_${epochs}_${batch_size}_${SUFFIX}

$PYTHON main.py \
    --root_path ${root_path} \
    --video_path ${video_path} \
    --annotation_path ${annotation_path} \
    --result_path ${result_path} \
    --dataset ${dataset} \
    --n_classes 101 \
    --n_pretrain_classes 1039 \
    --pretrain_path ${pretrain_path} \
    --ft_begin_module fc \
    --model resnet \
    --model_depth ${model_depth} \
    --batch_size ${batch_size} \
    --n_epochs ${epochs} \
    --n_threads 4 \
    --checkpoint 5
