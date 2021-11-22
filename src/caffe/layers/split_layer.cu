#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

/*template <typename Dtype>
__global__ void GradientSignCUDA(const int n, const Dtype* diff1,
    const Dtype* diff2, Dtype* indicator) {
  CUDA_KERNEL_LOOP(index, n) {
    if ((diff1[index] >= 0 && diff2[index] >= 0) || (diff1[index] < 0 && diff2[index] < 0)) {
      indicator[index] = 1;
    }
    else {
      indicator[index] = 0;
    }
  }
}

template <typename Dtype>
Dtype GradientSignGPU(const int n, const Dtype* diff1,
    const Dtype* diff2, Dtype* indicator) {
  GradientSignCUDA<Dtype><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS>>>(
      n, diff1, diff2, indicator);
  Dtype positive_num;
  caffe_gpu_asum(n, indicator, &positive_num);
  return positive_num / n;
}

template <typename Dtype>
Dtype GradientCorrelationGPU(const int n, const Dtype* diff1,
    const Dtype* diff2) {
  vector<int> sum_mult_shape(1, n);
  Blob<Dtype> sum_multiplier_(sum_mult_shape);
  Dtype* mult_data = sum_multiplier_.mutable_gpu_data();
  caffe_gpu_set(n, Dtype(1), mult_data);
  Dtype e1, e2, e1x2, e1p2, e2p2;
  caffe_gpu_dot(n, mult_data, diff1, &e1);
  caffe_gpu_dot(n, mult_data, diff2, &e2);
  caffe_gpu_dot(n, diff1, diff2, &e1x2);
  caffe_gpu_dot(n, diff1, diff1, &e1p2);
  caffe_gpu_dot(n, diff2, diff2, &e2p2);
  Dtype s12 = e1x2 / n - (e1 / n) * (e2 / n);
  Dtype s1  = e1p2 / n - (e1 / n) * (e1 / n);
  Dtype s2  = e2p2 / n - (e2 / n) * (e2 / n);
  return s12 / (sqrt(s1) * sqrt(s2) + 1e-6);
}*/

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    return;
  }
  //Dtype* indicator_data = indicator_.mutable_gpu_data();
  caffe_gpu_add(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                bottom[0]->mutable_gpu_diff());
  /*if (top.size() == 2) {
    const Dtype* top_diff0 = top[0]->gpu_diff();
    const Dtype* top_diff1 = top[1]->gpu_diff();
    //Layer<Dtype>::split_layer_positive_rate = GradientSignGPU(count_, top_diff0, top_diff1, indicator_data);
    //Layer<Dtype>::split_layer_negative_rate = 1 - Layer<Dtype>::split_layer_positive_rate;
    Layer<Dtype>::split_layer_correlation = GradientCorrelationGPU(count_, top_diff0, top_diff1);
  }*/
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    /*if (i == top.size() - 1) {
      //Layer<Dtype>::split_layer_positive_rate = GradientSignGPU(count_, bottom_diff, top_diff, indicator_data);
      //Layer<Dtype>::split_layer_negative_rate = 1 - Layer<Dtype>::split_layer_positive_rate;
      Layer<Dtype>::split_layer_correlation = GradientCorrelationGPU(count_, bottom_diff, top_diff);
    }*/
    caffe_gpu_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
