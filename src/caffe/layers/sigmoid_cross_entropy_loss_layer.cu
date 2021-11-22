#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossCountGPU(const int nthreads,
          const Dtype* target, Dtype* counts1, Dtype* counts2) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    if (target[i] == 1) {
      counts1[i] = 1;
      counts2[i] = 0;
    }
    else if (target[i] == 0) {
      counts1[i] = 0;
      counts2[i] = 1;
    }
    else {
      counts1[i] = 0;
      counts2[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          Dtype alpha, Dtype beta) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 1) {
      loss[i] = -alpha * 1.0 * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    }
    else if (target_value == 0) {
      loss[i] = -beta * 1.1 * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    }
    else {
      loss[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyLossNormlizeDiffGPU(const int nthreads,
    const Dtype* target, Dtype* diff, Dtype alpha, Dtype beta) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 1) {
      diff[i] *= alpha * 1.0;
    }
    else if (target_value == 0) {
      diff[i] *= beta * 1.1;
    }
    else {
      diff[i] *= 0;
    }
  }
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->count(2);
  CHECK_EQ(bottom[0]->shape(1), 1) << "Ground truth must be single channel.";
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();

  SigmoidCrossEntropyLossCountGPU<<<CAFFE_GET_BLOCKS(dim),
      CAFFE_CUDA_NUM_THREADS>>>(dim, target, loss_data, count_data);

  Dtype loss = 0;
  Dtype temp_loss;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  for (int i = 0; i < num; i++) {
    caffe_gpu_asum(dim, loss_data + i*dim, &count_pos);
    caffe_gpu_asum(dim, count_data + i*dim, &count_neg);
    Dtype pos_ratio = count_pos / (count_pos + count_neg);
    Dtype neg_ratio = count_neg / (count_pos + count_neg);

    SigmoidCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(dim),
        CAFFE_CUDA_NUM_THREADS>>>(dim, input_data + i*dim,
        target + i*dim, loss_data + i*dim, neg_ratio, pos_ratio);
    caffe_gpu_asum(dim, loss_data + i*dim, &temp_loss);
    loss += temp_loss;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyLossLayer<Dtype>::Backward_gpu(
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
    const int dim = bottom[0]->count(2);
    CHECK_EQ(bottom[0]->shape(1), 1) << "Ground truth must be single channel.";
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype* count_data = bottom[1]->mutable_gpu_diff();

    SigmoidCrossEntropyLossCountGPU<<<CAFFE_GET_BLOCKS(dim),
        CAFFE_CUDA_NUM_THREADS>>>(dim, target, bottom_diff, count_data);

    Dtype count_pos = 0;
    Dtype count_neg = 0;
    for (int i = 0; i < num; i++) {
      caffe_gpu_asum(dim, bottom_diff + i*dim, &count_pos);
      caffe_gpu_asum(dim, count_data + i*dim, &count_neg);
      Dtype pos_ratio = count_pos / (count_pos + count_neg);
      Dtype neg_ratio = count_neg / (count_pos + count_neg);

      caffe_gpu_sub(dim, sigmoid_output_data + i*dim, target + i*dim, bottom_diff + i*dim);

      SigmoidCrossEntropyLossNormlizeDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(dim),
          CAFFE_CUDA_NUM_THREADS>>>(dim, target + i*dim, bottom_diff + i*dim,
            neg_ratio, pos_ratio);
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyLossLayer);

}  // namespace caffe
