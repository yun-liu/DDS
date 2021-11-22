#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
  //indicator_.Reshape(bottom[0]->shape());
}

template <typename Dtype>
void SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    top[i]->ShareData(*bottom[0]);
  }
}

/*template <typename Dtype>
Dtype GradientSignCPU(const int n, const Dtype* diff1,
    const Dtype* diff2, Dtype* indicator) {
  for (int i = 0; i < n; ++i) {
    if ((diff1[i] >= 0 && diff2[i] >= 0) || (diff1[i] < 0 && diff2[i] < 0)) {
      indicator[i] = 1;
    }
    else {
      indicator[i] = 0;
    }
  }
  Dtype positive_num = caffe_cpu_asum(n, indicator);
  return positive_num / n;
}

template <typename Dtype>
Dtype GradientCorrelationCPU(const int n, const Dtype* diff1,
    const Dtype* diff2) {
  vector<int> sum_mult_shape(1, n);
  Blob<Dtype> sum_multiplier_(sum_mult_shape);
  Dtype* mult_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(n, Dtype(1), mult_data);
  Dtype e1, e2, e1x2, e1p2, e2p2;
  e1   = caffe_cpu_dot(n, mult_data, diff1);
  e2   = caffe_cpu_dot(n, mult_data, diff2);
  e1x2 = caffe_cpu_dot(n, diff1, diff2);
  e1p2 = caffe_cpu_dot(n, diff1, diff1);
  e2p2 = caffe_cpu_dot(n, diff2, diff2);
  Dtype s12 = e1x2 / n - (e1 / n) * (e2 / n);
  Dtype s1  = e1p2 / n - (e1 / n) * (e1 / n);
  Dtype s2  = e2p2 / n - (e2 / n) * (e2 / n);
  return s12 / (sqrt(s1) * sqrt(s2) + 1e-6);
}*/

template <typename Dtype>
void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    return;
  }
  //Dtype* indicator_data = indicator_.mutable_cpu_data();
  caffe_add(count_, top[0]->cpu_diff(), top[1]->cpu_diff(),
            bottom[0]->mutable_cpu_diff());
  /*if (top.size() == 2) {
    const Dtype* top_diff0 = top[0]->cpu_diff();
    const Dtype* top_diff1 = top[1]->cpu_diff();
    //Layer<Dtype>::split_layer_positive_rate = GradientSignCPU(count_, top_diff0, top_diff1, indicator_data);
    //Layer<Dtype>::split_layer_negative_rate = 1 - Layer<Dtype>::split_layer_positive_rate;
    Layer<Dtype>::split_layer_correlation = GradientCorrelationCPU(count_, top_diff0, top_diff1);
  }*/
  // Add remaining top blob diffs.
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    /*if (i == top.size() - 1) {
      //Layer<Dtype>::split_layer_positive_rate = GradientSignCPU(count_, bottom_diff, top_diff, indicator_data);
      //Layer<Dtype>::split_layer_negative_rate = 1 - Layer<Dtype>::split_layer_positive_rate;
      Layer<Dtype>::split_layer_correlation = GradientCorrelationCPU(count_, bottom_diff, top_diff);
    }*/
    caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SplitLayer);
#endif

INSTANTIATE_CLASS(SplitLayer);
REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
