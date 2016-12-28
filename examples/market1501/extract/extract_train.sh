set -e
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
base_model=caffenet
feature_name=fc7
model_file=./models/market1501/$base_model/snapshot/${base_model}.full_iter_18000.caffemodel

python examples/market1501/testing/extract_feature.py \
	examples/market1501/lists/train.lst \
	examples/market1501/datamat/train.lst.fc7.mat \
	examples/market1501/datamat/train.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/$base_model/dev.proto \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606
