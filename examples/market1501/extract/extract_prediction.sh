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
	examples/market1501/lists/test.lst \
	examples/market1501/datamat/test.lst.fc7.mat \
	examples/market1501/datamat/test.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/caffenet/dev.proto \
	--feature_name fc7 \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606

python examples/market1501/testing/extract_feature.py \
	examples/market1501/lists/query.lst \
	examples/market1501/datamat/query.lst.fc7.mat \
	examples/market1501/datamat/query.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/caffenet/dev.proto \
	--feature_name fc7 \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606
