#include <vector>

#include "caffe/layers/multi_label_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MultiLabelLossCountGPU(const int nthreads, const Dtype* target,
          Dtype* counts, const int cha, const int dim) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    int n = i / dim;
    int d = i % dim;
    bool pflag = false, nflag = false;
    for (int k = 0; k < cha; k++) {
      const int target_value = static_cast<int>(target[n*cha*dim + k*dim + d]);
      pflag = pflag || (target_value == 1);
      nflag = nflag || (target_value == 0);
    }
    if (pflag) {
      counts[n*cha*dim + 0*dim + d] = 1;
      counts[n*cha*dim + 1*dim + d] = 0;
    }
    else if (nflag) {
      counts[n*cha*dim + 0*dim + d] = 0;
      counts[n*cha*dim + 1*dim + d] = 1;
    }
    else {
      counts[n*cha*dim + 0*dim + d] = 0;
      counts[n*cha*dim + 1*dim + d] = 0;
    }
  }
}

template <typename Dtype>
__global__ void MultiLabelLossForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          Dtype alpha, Dtype beta) {
  CUDA_KERNEL_LOOP (i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 1) {
      loss[i] = alpha * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    }
    else if (target_value == 0) {
      loss[i] = beta * (input_data[i] * (target[i] - (input_data[i] >= 0)) -
        log(1 + exp(input_data[i] - 2 * input_data[i] * (input_data[i] >= 0))));
    }
  }
}

template <typename Dtype>
__global__ void MultiLabelLossNormlizeDiffGPU(const int nthreads,
    const Dtype* target, Dtype* diff, Dtype alpha, Dtype beta) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == 1) {
      diff[i] *= alpha;
    }
    else if (target_value == 0) {
      diff[i] *= beta;
    }
    else {
      diff[i] *= 0;
    }
  }
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_gpu(
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
  Dtype* count_data = bottom[1]->mutable_gpu_diff();

  MultiLabelLossCountGPU<Dtype><<<CAFFE_GET_BLOCKS(num*dim),
      CAFFE_CUDA_NUM_THREADS>>>(num*dim, target, count_data, cha, dim);

  Dtype loss = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  for (int i = 0; i < num; i++) {
    caffe_gpu_asum(dim, count_data + i*cha*dim + 0*dim, &count_pos);
    caffe_gpu_asum(dim, count_data + i*cha*dim + 1*dim, &count_neg);
    Dtype pos_ratio = count_pos / (count_pos + count_neg);
    Dtype neg_ratio = count_neg / (count_pos + count_neg);

    MultiLabelLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(cha*dim),
        CAFFE_CUDA_NUM_THREADS>>>(cha*dim, input_data + i*cha*dim,
        target + i*cha*dim, loss_data + i*cha*dim, neg_ratio, pos_ratio);
    Dtype temp_loss;
    caffe_gpu_asum(cha*dim, loss_data + i*cha*dim, &temp_loss);
    loss += temp_loss;
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_gpu(
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
    Dtype* count_data = bottom[1]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);

    Dtype count_pos = 0;
    Dtype count_neg = 0;
    for (int i = 0; i < num; i++) {
      caffe_gpu_asum(dim, count_data + i*cha*dim + 0*dim, &count_pos);
      caffe_gpu_asum(dim, count_data + i*cha*dim + 1*dim, &count_neg);
      Dtype pos_ratio = count_pos / (count_pos + count_neg);
      Dtype neg_ratio = count_neg / (count_pos + count_neg);

      MultiLabelLossNormlizeDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(cha*dim),
          CAFFE_CUDA_NUM_THREADS>>>(cha*dim, target + i*cha*dim,
          bottom_diff + i*cha*dim, neg_ratio, pos_ratio);
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_gpu_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiLabelLossLayer);

}  // namespace caffe
