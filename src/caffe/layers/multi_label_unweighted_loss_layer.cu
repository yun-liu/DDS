#include <vector>

#include "caffe/layers/multi_label_unweighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiLabelUnweightedLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 1) {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
    else if (target_value == 0) {
      loss[i] = input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
  }
}

template <typename Dtype>
__global__ void MultiLabelUnweightedLossNormlizeDiffGPU(const int nthreads,
    const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value != 1 && target_value != 0) {
      diff[i] = 0;
    }
  }
}

template <typename Dtype>
void MultiLabelUnweightedLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int num = bottom[0]->num();
  const int cha = bottom[0]->shape(1);
  const int dim = bottom[0]->count(2);
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype loss = 0;
  for (int i = 0; i < num; i++) {
    MultiLabelUnweightedLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(cha*dim),
        CAFFE_CUDA_NUM_THREADS>>>(cha*dim, input_data + i*cha*dim,
        target + i*cha*dim, loss_data + i*cha*dim);
    Dtype temp_loss;
    caffe_gpu_asum(cha*dim, loss_data + i*cha*dim, &temp_loss);
    loss += temp_loss;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultiLabelUnweightedLossLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[0]->num();
    const int cha = bottom[0]->shape(1);
    const int dim = bottom[0]->count(2);
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);

    for (int i = 0; i < num; i++) {
      MultiLabelUnweightedLossNormlizeDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(cha*dim),
          CAFFE_CUDA_NUM_THREADS>>>(cha*dim, target + i*cha*dim,
          bottom_diff + i*cha*dim);
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiLabelUnweightedLossLayer);

}  // namespace caffe
