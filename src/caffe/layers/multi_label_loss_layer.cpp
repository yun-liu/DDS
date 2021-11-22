#include <vector>

#include "caffe/layers/multi_label_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
        "MULTI_LABEL_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int num = bottom[0]->num();
  const int cha = bottom[0]->shape(1);
  const int dim = bottom[0]->count(2);
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype temp_loss_pos = 0;
  Dtype temp_loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;

  for (int i = 0; i < num; i++) {
    count_pos = 0;
    count_neg = 0;
    for (int k = 0; k < dim; k++) {
      bool pflag = false, nflag = false;;
      for (int j = 0; j < cha; j++) {
        pflag = pflag || (target[i*cha*dim + j*dim + k] == 1);
        nflag = nflag || (target[i*cha*dim + j*dim + k] == 0);
      }
      if (pflag) {
        count_pos++;
      }
      else if (nflag) {
        count_neg++;
      }
    }

    for (int j = 0; j < cha; j++) {
      temp_loss_pos = 0;
      temp_loss_neg = 0;
      for (int k = 0; k < dim; k++) {
        int idx = i*cha*dim + j*dim + k;
        if (target[idx] == 1) {
          temp_loss_pos -= input_data[idx] * (target[idx] - (input_data[idx] >= 0)) -
            log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
        }
        else if (target[idx] == 0) {
          temp_loss_neg -= input_data[idx] * (target[idx] - (input_data[idx] >= 0)) -
            log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
        }
      }
      loss_pos += temp_loss_pos * count_neg / (count_pos + count_neg);
      loss_neg += temp_loss_neg * count_pos / (count_pos + count_neg);
    }
  }
  top[0]->mutable_cpu_data()[0] = (loss_pos + loss_neg) / num;
}

template <typename Dtype>
void MultiLabelLossLayer<Dtype>::Backward_cpu(
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
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);

    Dtype count_pos = 0;
    Dtype count_neg = 0;
    for (int i = 0; i < num; i++) {
      count_pos = 0;
      count_neg = 0;
      for (int k = 0; k < dim; k++) {
        bool pflag = false, nflag = false;;
        for (int j = 0; j < cha; j++) {
          pflag = pflag || (target[i*cha*dim + j*dim + k] == 1);
          nflag = nflag || (target[i*cha*dim + j*dim + k] == 0);
        }
        if (pflag) {
          count_pos++;
        }
        else if (nflag) {
          count_neg++;
        }
      }

      for (int j = 0; j < cha; j++) {
        for (int k = 0; k < dim; k++) {
          if (target[i*cha*dim+j*dim+k] == 1) {
            bottom_diff[i*cha*dim+j*dim+k] *= count_neg / (count_pos + count_neg);
          }
          else if (target[i*cha*dim+j*dim+k] == 0) {
            bottom_diff[i*cha*dim+j*dim+k] *= count_pos / (count_pos + count_neg);
          }
          else
            bottom_diff[i*cha*dim+j*dim+k] *= 0;
        }
      }
    }

    // Scale down gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(count, loss_weight / num, bottom_diff);
  }
}

INSTANTIATE_CLASS(MultiLabelLossLayer);
REGISTER_LAYER_CLASS(MultiLabelLoss);

}  // namespace caffe
