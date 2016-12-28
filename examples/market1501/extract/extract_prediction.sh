set -e
if [ ! -n "$1" ] ;then
    echo "\$1 is empty, default is 0"
    gpu=0
else
    echo "use $1-th gpu"
    gpu=$1
fi
#base_model=caffenet
#base_model=vgg16
base_model=googlenet
#feature_name=fc7
feature_name=pool5/7x7_s1
#feature_name=pool5
model_file=./models/market1501/$base_model/snapshot/${base_model}.full_iter_18000.caffemodel

python examples/market1501/testing/extract_feature.py \
	examples/market1501/lists/test.lst \
	examples/market1501/datamat/test.lst.fc7.mat \
	examples/market1501/datamat/test.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/$base_model/dev.proto \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606

python examples/market1501/testing/extract_feature.py \
	examples/market1501/lists/query.lst \
	examples/market1501/datamat/query.lst.fc7.mat \
	examples/market1501/datamat/query.lst.score.mat \
	--gpu $gpu \
	--model_def ./models/market1501/$base_model/dev.proto \
	--feature_name $feature_name \
	--pretrained_model $model_file \
	--mean_value 97.8286,99.0468,105.606
