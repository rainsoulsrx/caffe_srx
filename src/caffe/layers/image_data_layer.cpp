#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <time.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"

#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

//add by shenruixue
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
//end by shenruixue


namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
}

//added by shenruixue 20180227
template <typename Dtype>
void ImageDataLayer<Dtype>::CutOutImg(Mat &src, Mat &dst)
{
    dst = src.clone();
    int height = src.rows;
    int width = src.cols;
    const float MAX_RATIO = 0.9;
    const float MMIN_RATIO = 0.2;
    int current_iter = Caffe::current_iter();
    int max_iter = Caffe::max_iter();
//    LOG(INFO) << "current_iter is" << current_iter;
//    LOG(INFO) << "max_iter is" << max_iter;
    float cur_ratio = MMIN_RATIO + MAX_RATIO * float(current_iter) / float(max_iter);
    if(max_iter < 10)
    {
        cur_ratio = MMIN_RATIO;
    }
    int cut_out_len = (int)(width * cur_ratio);
    int x_start = 0;
    int x_end   = width - cut_out_len;
    int y_start = 0;
    int y_end   = height - cut_out_len;
    srand((unsigned)time(NULL));
    int x = (rand() % (x_end - x_start))+ x_start + 1;
    int y = (rand() % (y_end - y_start))+ y_start + 1;
//    LOG(INFO) << "cur_ratio is " << cur_ratio;
//    LOG(INFO) << "cur_ratio is" << cur_ratio;
//    LOG(INFO) << "cur_ratio is" << cur_ratio;
    cv::Mat Roi(dst, cv::Rect(x, y, cut_out_len, cut_out_len));
    cv::RNG rnger(cv::getTickCount());
    rnger.fill(Roi, cv::RNG::UNIFORM, cv::Scalar::all(0), cv::Scalar::all(256));
    //Roi = cv::Scalar(rng.uniform(0,255),rng.uniform(0,255),rng.uniform(0,255));

}
//added by shenruixue 20180227


template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();
  //add by shenruixue
  const int num_labels = this->layer_param_.image_data_param().num_labels();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos;
  //modified by shenruixue
  //int label;
  vector <float> labels;
  LOG(INFO) << "num_labels "<<num_labels;
  //modified by shenruixue

  //added by shenruixue
  float label=0;
  string imagename;
  int num =0;
  while (infile>>imagename)
  {
      num =0;
      labels.clear();
      while(infile>>label)
      {
          num++;
          labels.push_back(label);
          if (num==num_labels)
          {
              lines_.push_back(std::make_pair(imagename, labels));
              break;
          }
      }
  }
  //added by shenruixue

//  while (std::getline(infile, line)) {
//    pos = line.find_last_of(' ');
//    label = atoi(line.substr(pos + 1).c_str());
//    lines_.push_back(std::make_pair(line.substr(0, pos), label));
//  }
  lines_ori = lines_;
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(INFO) << "lines_ori A total of " << lines_ori.size() << " images.";
  CHECK(!lines_.empty()) << "File is empty";

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  } else {
    if (this->phase_ == TRAIN && Caffe::solver_rank() > 0 &&
        this->layer_param_.image_data_param().rand_skip() == 0) {
      LOG(WARNING) << "Shuffling or skipping recommended for multi-GPU";
    }
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  LOG(INFO) << "lines_ori A total of " << lines_ori.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
//  // label
//  vector<int> label_shape(1, batch_size);
//  top[1]->Reshape(label_shape);
  //modified by shenruixue
  top[1]->Reshape(batch_size,num_labels,1,1);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
//    this->prefetch_[i]->label_.Reshape(label_shape);
      this->prefetch_[i]->label_.Reshape(batch_size,num_labels,1,1);
  }

}

template <typename Dtype>
int ImageDataLayer<Dtype>::MyRandomData (int i) { 
    return caffe_rng_rand()%i;
}

//add by shenruixue 20171208
template <typename Dtype>
void ImageDataLayer<Dtype>::Shuffle_ori_inner()
{
  lines_shuffle.clear();
  const int lines_size = lines_ori.size();
  vector<vector<std::pair<std::string, vector<float> > > > vv_lines_ori;
  vector<std::pair<std::string, vector<float> > > v_tmp;
  int label_ori = 0;
  for (int i = 0; i < lines_size; i++)
  {
    string line = lines_ori[i].first;
    vector<float> label = lines_ori[i].second;
    if(i == lines_size - 1)
    {
      v_tmp.push_back(std::make_pair(line, label));
      vv_lines_ori.push_back(v_tmp);
    }
    else if(label[0] == label_ori)
    {
      v_tmp.push_back(std::make_pair(line, label));
      label_ori = label[0];
    }
    else
    {
      vv_lines_ori.push_back(v_tmp);
      v_tmp.clear();
      v_tmp.push_back(std::make_pair(line, label));
      label_ori = label[0];
    }
  }

  const int vv_lines_ori_size = vv_lines_ori.size();
  for(int i = 0; i < vv_lines_ori_size; i++)
  {
    std::random_shuffle(vv_lines_ori[i].begin(), vv_lines_ori[i].end(), ImageDataLayer<Dtype>::MyRandomData);
  }
//   ofstream outf;
//   outf.open("abc.txt");
  for(int i = 0; i < vv_lines_ori_size; i ++)
  {
    for(int j = 0; j < vv_lines_ori[i].size(); j++)
    {
      lines_shuffle.push_back(std::make_pair(vv_lines_ori[i][j].first, vv_lines_ori[i][j].second));
//      for(int k = 0; k < (vv_lines_ori[i][j].second).size(); k++)
//      {
//          ostringstream oss;  //创建一个格式化输出流
//          oss<<((vv_lines_ori[i][j].second)[k]);             //把值传递如流中
//          string tmp = oss.str();
//          outf << vv_lines_ori[i][j].first << " " << tmp << endl;
//      }

    }
  }
//  outf.close();
}


template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
   /*caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);*/
  lines_.clear();
  Shuffle_ori_inner();
  lines_ = lines_shuffle;
  const int num_images = lines_.size();
  DLOG(INFO) << "My Shuffle.";
  vector<std::pair<std::string, vector<float> > > tlines_;
  vector<int> tnum;
  int pairsize = this->layer_param_.image_data_param().pair_size();

  for(int i = 0; i < num_images / pairsize; i ++)
  {
	  tnum.push_back(i);
  }
  std::random_shuffle(tnum.begin(), tnum.end(), ImageDataLayer<Dtype>::MyRandomData);
  tlines_.clear();
  for(int i = 0; i < num_images / pairsize; i ++)
  {
	  for(int j = 0; j < pairsize; j ++)
	  {
		  tlines_.push_back(lines_[tnum[i] * pairsize + j]);
	  }
  }
  lines_ = tlines_;

}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();
  //added by shenruixue
  const int num_labels = image_data_param.num_labels();
  const int object_scale = image_data_param.object_scale();
  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
        new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    read_time += timer.MicroSeconds();
    timer.Start();

    //added by shenruixue 20171110
    cv::Mat cv_img_resize_dst;
    if (object_scale>0)
    {
      float object_scale_rand = float(rand()%(object_scale+1)+10)/10;
      int resize_width = int(object_scale_rand* new_width);
      int resize_height = int(object_scale_rand* new_height);
      cv::Mat cv_img_resize(resize_height,resize_width,CV_8UC3,Scalar(128,128,128));
      int shift_y = (resize_height-new_height)/2;
      int shift_x = (resize_width-new_width)/2;

      cv_img.copyTo(cv_img_resize(Rect(shift_x, shift_y, new_width, new_height)));

      resize(cv_img_resize,cv_img_resize_dst,Size(new_width, new_height));

    }
    else
    {
      cv_img.copyTo(cv_img_resize_dst);
    }
    //end added by shenruixue
    //add by shenruixue 20180226
    cv::Mat cv_img_distort_dst;
    this->data_transformer_->DistortImage(cv_img_resize_dst, cv_img_distort_dst);
    const bool is_cut = image_data_param.is_cut();
    cv::Mat cv_img_cut_dst;
    if(is_cut)
    {
        CutOutImg(cv_img_distort_dst, cv_img_cut_dst);
    }
    else
    {
        cv_img_distort_dst.copyTo(cv_img_cut_dst);
    }


//    string str = std::to_string(read_time);
//    cv::imwrite("/hard_disk2/100.202_files/ReID/gg_net_triloss/local_fea/test/" + str + ".jpg", cv_img_cut_dst);

    // Apply transformations (mirror, crop...) to the image
    int offset = batch->data_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + offset);
    this->data_transformer_->Transform(cv_img_cut_dst, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();
//modified by shenruixue
    //prefetch_label[item_id] = lines_[lines_id_].second;
    for (int num_labels_i=0;num_labels_i<num_labels;num_labels_i++)
    {
        prefetch_label[item_id*num_labels+num_labels_i] = lines_[lines_id_].second[num_labels_i];
    }
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";


}


INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
