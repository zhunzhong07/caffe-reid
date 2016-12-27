set -e
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
model_file=./models/market1501/caffenet/snapshot/caffenet.full_iter_18000.caffemodel

python examples/market1501/testing/extract_feature.py \
	examples/market1501/lists/train.lst \
	examples/market1501/datamat/train.lst.fc7.mat \
	examples/market1501/datamat/train.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/caffenet/dev.proto \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606
