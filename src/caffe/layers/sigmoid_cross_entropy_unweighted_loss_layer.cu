#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_unweighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 0 || target_value == 1) {
      loss[i] = -input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0)));
    }
    else {
      loss[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossNormlizeDiffGPU(const int nthreads,
    const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value != 0 && target_value != 1) {
      diff[i] = 0;
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyUnweightedLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  CHECK_EQ(bottom[0]->shape(1), 1) << "Ground truth must be single channel.";
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();

  Dtype loss = 0;
  SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data);
  caffe_gpu_asum(count, loss_data, &loss);
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyUnweightedLossLayer<Dtype>::Backward_gpu(
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
    CHECK_EQ(bottom[0]->shape(1), 1) << "Ground truth must be single channel.";
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

    caffe_gpu_sub(count, sigmoid_output_data, target, bottom_diff);
    SigmoidCrossEntropyLossNormlizeDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, target, bottom_diff);

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyUnweightedLossLayer);

}  // namespace caffe
