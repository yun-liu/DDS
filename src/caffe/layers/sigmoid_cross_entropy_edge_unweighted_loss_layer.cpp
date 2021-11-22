#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_edge_unweighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidCrossEntropyEdgeUnweightedLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyEdgeUnweightedLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->num()*bottom[1]->count(2)) <<
      "SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidCrossEntropyEdgeUnweightedLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int num = bottom[1]->num();
  const int cha = bottom[1]->shape(1);
  const int dim = bottom[1]->count(2);
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0;

  Blob<Dtype> edge_map;
  edge_map.Reshape(num, 1, bottom[0]->height(), bottom[0]->width());
  Dtype* label = edge_map.mutable_cpu_data();

  for (int i = 0; i < num; i++) {
    for (int j = 0; j < dim; j++) {
      bool pflag = false, nflag = false;;
      for (int k = 0; k < cha; k++) {
        pflag = pflag || (target[i*cha*dim + k*dim + j] == 1);
        nflag = nflag || (target[i*cha*dim + k*dim + j] == 0);
      }
      if (pflag) {
        label[i*dim+j] = Dtype(1);
      }
      else if (nflag) {
        label[i*dim+j] = Dtype(0);
      }
      else {
        label[i*dim+j] = Dtype(255);
      }
    }

    for (int j = 0; j < dim; j++) {
      const int idx = i*dim + j;
      if (label[idx] == 1) {
        loss -= input_data[idx] * (label[idx] - (input_data[idx] >= 0)) -
          log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      }
      else if (label[idx] == 0) {
        loss -= input_data[idx] * (label[idx] - (input_data[idx] >= 0)) -
          log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}

template <typename Dtype>
void SigmoidCrossEntropyEdgeUnweightedLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const int num = bottom[1]->num();
    const int cha = bottom[1]->shape(1);
    const int dim = bottom[1]->count(2);
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    Blob<Dtype> edge_map;
    edge_map.Reshape(num, 1, bottom[0]->height(), bottom[0]->width());
    Dtype* label = edge_map.mutable_cpu_data();

    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < dim; j++) {
        bool pflag = false, nflag = false;;
        for (int k = 0; k < cha; k++) {
          pflag = pflag || (target[i*cha*dim + k*dim + j] == 1);
          nflag = nflag || (target[i*cha*dim + k*dim + j] == 0);
        }
        if (pflag) {
          label[i*dim+j] = 1;
        }
        else if (nflag) {
          label[i*dim+j] = 0;
        }
        else {
          label[i*dim+j] = 255;
        }
      }
    }

    caffe_sub(count, sigmoid_output_data, label, bottom_diff);
    for (int i = 0; i < num; ++i) {
    	for (int j = 0; j < dim; j++) {
        const int target_value = static_cast<int>(label[i*dim + j]);
      	if (target_value != 1 && target_value != 0) {
          bottom_diff[i*dim + j] = 0;
      	}
     	}
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(SigmoidCrossEntropyEdgeUnweightedLossLayer);
REGISTER_LAYER_CLASS(SigmoidCrossEntropyEdgeUnweightedLoss);

}  // namespace caffe
