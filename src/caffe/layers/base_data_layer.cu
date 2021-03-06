#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

template <typename Dtype>
void ReidPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ReidBatch<Dtype>* batch = this->prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  CHECK_EQ(top[0]->count(), batch->data_.count()*2);
  top[0]->Reshape(batch->data_.num()*2, batch->data_.channels(), batch->data_.height(), batch->data_.width());
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(), top[0]->mutable_gpu_data());
  caffe_copy(batch->datap_.count(), batch->datap_.gpu_data(), top[0]->mutable_gpu_data()+batch->data_.count());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    // Reshape to loaded labels.
    vector<int> shape = batch->label_.shape();
    CHECK_LT(shape.size(), 2);
    CHECK_EQ(top[1]->count(), batch->label_.count()*2);
    shape[0] *= 2;
    top[1]->Reshape(shape);
    // Copy the labels.
    caffe_copy(batch->label_.count(),  batch->label_.gpu_data(),  top[1]->mutable_gpu_data());
    caffe_copy(batch->labelp_.count(), batch->labelp_.gpu_data(), top[1]->mutable_gpu_data()+batch->label_.count());
  }

  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  prefetch_free_.push(batch);
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ReidPrefetchingDataLayer);

}  // namespace caffe
