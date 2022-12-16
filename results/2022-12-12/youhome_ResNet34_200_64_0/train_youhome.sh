PYTHON="/home/aperture/miniconda3/envs/torch/bin/python"

DATE=`date +%Y-%m-%d`

epochs=200
model_depth=34
batch_size=64
n_classes=31

dataset=youhome

root_path="/media/aperture/Delta/Descartes/mp4data_2021_frames"
video_path="img"
annotation_path="anno/YouHome.json"
result_path="results"
# pretrain_path="pretrain/r3d18_KM_200ep.pth"
# pretrain_path="pretrain/r3d50_KMS_200ep.pth"
pretrain_path="pretrain/r3d34_KM_200ep.pth"
# pretrain_path="pretrain/r3d101_KM_200ep.pth"

SUFFIX=0

if [ ! -d "$DIRECTORY" ]; then
    mkdir ${root_path}/${result_path}/${DATE}
fi

mkdir ${root_path}/${result_path}/${DATE}/${dataset}_ResNet${model_depth}_${epochs}_${batch_size}_${SUFFIX}

cp train_youhome.sh ${root_path}/${result_path}/${DATE}/${dataset}_ResNet${model_depth}_${epochs}_${batch_size}_${SUFFIX}

$PYTHON main.py \
    --root_path ${root_path} \
    --video_path ${video_path} \
    --annotation_path ${annotation_path} \
    --result_path ${result_path}/${DATE}/${dataset}_ResNet${model_depth}_${epochs}_${batch_size}_${SUFFIX} \
    --dataset ${dataset} \
    --n_classes ${n_classes} \
    --learning_rate 0.1 \
    --multistep_milestones 20 50 100 150\
    --n_pretrain_classes 1039 \
    --pretrain_path ${pretrain_path} \
    --ft_begin_module fc \
    --model resnet \
    --model_depth ${model_depth} \
    --batch_size ${batch_size} \
    --n_epochs ${epochs} \
    --n_threads 1 \
    --checkpoint 5
