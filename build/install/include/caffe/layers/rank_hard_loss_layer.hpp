#ifndef CAFFE_RANK_HARD_LOSS_LAYER_HPP_
#define CAFFE_RANK_HARD_LOSS_LAYER_HPP_

#include <vector>

#include <algorithm>
#include <cmath>
#include <cfloat>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
    
  template <typename Dtype>
  class RankHardLossLayer : public LossLayer<Dtype> {
  public:
    explicit RankHardLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                            const vector<Blob<Dtype>*>& top);
    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                         const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "RankHardLoss"; }
    virtual inline int ExactNumBottomBlobs() const { return 3; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
  
    static int MyRandom(int i);
    void set_mask(const vector<Blob<Dtype>*>& bottom);
    Dtype CalcuteShortestPath(Dtype* local_mat, int anchor, Dtype* index_local_data, int height_local);
  
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                             const vector<Blob<Dtype>*>& top);
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
                              
    Blob<Dtype> diff_ ;
    Blob<Dtype> dis_ ;
    Blob<Dtype> mask_ ;
    Blob<Dtype> dis_local_neg_ ;
    Blob<Dtype> dis_local_pos_ ;
    Blob<Dtype> index_local_neg_ ; //storage the index of shortest path
    Blob<Dtype> index_local_pos_ ;
    Blob<Dtype> total_dis_local_neg_ ;
    Blob<Dtype> total_dis_local_pos_ ;
  };

}

#endif  //CAFFE_RANK_HARD_LOSS_LAYER_HPP_
