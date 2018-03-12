#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/rank_hard_loss_layer.hpp"

using std::max;

using namespace std;
using namespace cv;


namespace caffe {
    

template <typename Dtype>
int RankHardLossLayer<Dtype>::MyRandom(int i)
{
    return caffe_rng_rand()%i;
}
 
template <typename Dtype>
void RankHardLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
          
    LossLayer<Dtype>::LayerSetUp(bottom, top);
    
    CHECK_EQ(bottom.size(), 3) << "RankHardLossLayer Layer takes three blobs as input.";

}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	LossLayer<Dtype>::Reshape(bottom, top);
    
    diff_.ReshapeLike(*bottom[0]);
    dis_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
    //2: one for most hardest pos and one for most hardest neg, each one dim is 7*7(bottom[2]->height()*bottom[2]->height())
    dis_local_neg_.Reshape(bottom[2]->num(), 1, bottom[2]->height(), bottom[2]->height());
    dis_local_pos_.Reshape(bottom[2]->num(), 1, bottom[2]->height(), bottom[2]->height());
    index_local_neg_.Reshape(bottom[2]->num(), 1, bottom[2]->height(), bottom[2]->height());
    index_local_pos_.Reshape(bottom[2]->num(), 1, bottom[2]->height(), bottom[2]->height());
    total_dis_local_pos_.Reshape(bottom[2]->num(), 1, 1, 1);
    total_dis_local_neg_.Reshape(bottom[2]->num(), 1, 1, 1);
    mask_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::set_mask(const vector<Blob<Dtype>*>& bottom)
{

	//RankParameter rank_param = this->layer_param_.rank_param();
	int neg_hard_num = this->layer_param_.rank_hard_loss_param().neg_num();
	int pos_hard_num = this->layer_param_.rank_hard_loss_param().pos_num();
    CHECK_EQ(neg_hard_num, pos_hard_num);
    //float margin = this->layer_param_.rank_hard_loss_param().margin();

	const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_data_local = bottom[2]->cpu_data();
	const Dtype* label = bottom[1]->cpu_data();
    //int count = bottom[0]->count();
	int num = bottom[0]->num();
    int num_local = bottom[2]->num();
    CHECK_EQ(num, num_local);
	int dim = bottom[0]->count() / bottom[0]->num();
    int dim_local = bottom[2]->count() / bottom[2]->num();
    int channel_local = bottom[2]->channels();
    int height_local = bottom[2]->height();
    //int width_local = bottom[2]->width();

	Dtype* dis_data = dis_.mutable_cpu_data();
    Dtype* dis_local_neg_data = dis_local_neg_.mutable_cpu_data();
    Dtype* dis_local_pos_data = dis_local_pos_.mutable_cpu_data();
    Dtype* index_local_neg_data = index_local_neg_.mutable_cpu_data();
    Dtype* index_local_pos_data = index_local_pos_.mutable_cpu_data();
    Dtype* total_dis_local_neg_data = total_dis_local_neg_.mutable_cpu_data();
    Dtype* total_dis_local_pos_data = total_dis_local_pos_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < num * num; i ++)
	{
		dis_data[i] = 0;
		mask_data[i] = 0;
	}
    for(int i = 0; i < num * 1 * height_local * height_local; i ++)
    {
        dis_local_neg_data[i] = 0;
        dis_local_pos_data[i] = 0;
        index_local_neg_data[i] = -1;
        index_local_pos_data[i] = -1;
    }
    for(int i = 0; i < num; i ++)
    {
        total_dis_local_neg_data[i] = 0;
        total_dis_local_pos_data[i] = 0;
    }

	// calculate distance
	for(int i = 0; i < num; i ++)
	{
		for(int j = i + 1; j < num; j ++)
		{
			const Dtype* fea1 = bottom_data + i * dim;
			const Dtype* fea2 = bottom_data + j * dim;
			Dtype ts = 0;
			for(int k = 0; k < dim; k ++)
			{
			  ts += (fea1[k] * fea2[k]) ;
			}
            dis_data[i * num + j] = 1.0 - ts;
            dis_data[j * num + i] = 1.0 - ts;
		}
	}

	//select samples
	vector<pair<float, int> >neg_hard_pairs;
	vector<pair<float, int> >pos_hard_pairs;
	vector<int> sid_neg;
	vector<int> sid_pos;

	//for each sample, find its poss and negs
	for(int i = 0; i < num; i ++)
	{
		neg_hard_pairs.clear();
		pos_hard_pairs.clear();
		sid_neg.clear();
		sid_pos.clear();
		for(int j = 0; j < num; j ++)
		{
			if(label[j] == label[i])
			{
				pos_hard_pairs.push_back(make_pair(dis_data[i * num + j], j));
			}
			else
			{
				neg_hard_pairs.push_back(make_pair(dis_data[i * num + j], j));
			}
			
		}
		//sort the vector to get the hardest neg and pos
		sort(neg_hard_pairs.begin(), neg_hard_pairs.end());
		sort(pos_hard_pairs.begin(), pos_hard_pairs.end());
//        for(int i = 0; i < neg_hard_pairs.size(); i++)
//        {
//            LOG(INFO) << "neg_hard_pairs.i  is" << i << " "<< neg_hard_pairs[i].first << " " << neg_hard_pairs[i].second;
//        }
//        for(int i = 0; i < pos_hard_pairs.size(); i++)
//        {
//            LOG(INFO) << "pos_hard_pairs.i  is" << i << " "<< pos_hard_pairs[i].first << " " << pos_hard_pairs[i].second;
//        }
		//push index of hardest pos and neg into vectors sid
        CHECK_LE(neg_hard_num, neg_hard_pairs.size());
        CHECK_LE(pos_hard_num, pos_hard_pairs.size());
        for(int i_neg_hard = 0; i_neg_hard < neg_hard_num; i_neg_hard ++)
		{
            sid_neg.push_back(neg_hard_pairs[i_neg_hard].second);
		}

        //LOG(INFO)<< "tmp_index is " << " " << tmp_index;
        //LOG(INFO) << "i is" << i;
        //LOG(INFO) << "pos_hard_pairs is" << pos_hard_pairs.size();
        for(int i_pos_hard = 0; i_pos_hard < pos_hard_num; i_pos_hard++)
		{
            //sid_pos.push_back(pos_hard_pairs[tmp_index - i].second);
            sid_pos.push_back(pos_hard_pairs.back().second);
            //LOG(INFO) << "pos_hard_pairs.front().second is" << pos_hard_pairs.front().second;
		}
        int index_anchor = i * height_local * height_local;
		//make correspoding mask to be 1
		for(int j = 0; j < min(neg_hard_num, (int)(sid_neg.size()) ); j ++)
		{
			mask_data[i * num + sid_neg[j]] = 1;
            //calculate the local distance
            const Dtype* fea1_local = bottom_data_local + i * dim_local;
            const Dtype* fea2_local = bottom_data_local + sid_neg[j] * dim_local;
            for(int i_local = 0; i_local < height_local; i_local++)
            {
                for(int j_local = 0; j_local < height_local; j_local++)
                {
                    Dtype ts_local_neg = 0;
                    Dtype norm2_i_j = 0;
                    for(int k_local = 0; k_local < channel_local; k_local++)
                    {
                        int index_tmp_fea1 = i_local * channel_local + k_local;
                        int index_tmp_fea2 = j_local * channel_local + k_local;
                        norm2_i_j += (fea1_local[index_tmp_fea1] - fea2_local[index_tmp_fea2]) * (fea1_local[index_tmp_fea1] - fea2_local[index_tmp_fea2]);//change here to be exp dis

                        //int index_tmp = j_local * channel_local + k_local;
                        //norm2_i_j += (fea1_local[index_tmp] - fea2_local[index_tmp]) * (fea1_local[index_tmp] - fea2_local[index_tmp]);//change here to be exp dis
                    }
                    //LOG(INFO) << "neg norm2_i_j is" << norm2_i_j;
                    //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                    if(norm2_i_j > 10.0)
                    {norm2_i_j = 10.0;}
                    ts_local_neg = (std::exp(norm2_i_j) - 1.0) / (std::exp(norm2_i_j) + 1.0);
                    dis_local_neg_data[index_anchor + i_local * height_local + j_local] = ts_local_neg;
                    //dis_local_neg_data[i * height_local * height_local + j_local * height_local + i_local] = ts_local_neg;
                }

            }

		}

        total_dis_local_neg_data[i] = CalcuteShortestPath(dis_local_neg_data, index_anchor, index_local_neg_data, height_local);
        //LOG(INFO) << "total_dis_local_neg_data is" << " " <<  " " << total_dis_local_neg_data[i];
        //make correspoding mask to be 2
		for(int j = 0; j < min(pos_hard_num, (int)(sid_pos.size()) ); j ++)
		{
            int tmp_index = pos_hard_pairs.size() - 1;
            if(tmp_index == 0)
            {
                //LOG(INFO) << "tmp_indexis 0";
                continue;
            }
            mask_data[i * num + sid_pos[j]] = 2;
            //calculate the local distance
            const Dtype* fea1_local = bottom_data_local + i * dim_local;
            const Dtype* fea2_local = bottom_data_local + sid_pos[j] * dim_local;
            //LOG(INFO)<< "sid_pos[j] is " << " " <<sid_pos[j];
            //LOG(INFO)<< "i is " << " " << i;
            //LOG(INFO)<< "pos_hard_pairs is " << " " << pos_hard_pairs.size();
            //LOG(INFO)<< "neg_hard_pairs is " << " " << neg_hard_pairs.size();
            for(int i_local = 0; i_local < height_local; i_local++)
            {
                for(int j_local = 0; j_local < height_local; j_local++)
                {
                    Dtype ts_local_pos = 0;
                    Dtype norm2_i_j = 0;
                    for(int k_local = 0; k_local < channel_local; k_local++)
                    {
                        int index_tmp_fea1 = i_local * channel_local + k_local;
                        int index_tmp_fea2 = j_local * channel_local + k_local;
                        //LOG(INFO)<< "fea1_local[index_tmp] is " << " " << fea1_local[index_tmp];
                        //LOG(INFO)<< "fea2_local[index_tmp] is " << " " << fea2_local[index_tmp];
                        norm2_i_j += (fea1_local[index_tmp_fea1] - fea2_local[index_tmp_fea2]) * (fea1_local[index_tmp_fea1] - fea2_local[index_tmp_fea2]);//change here to be exp dis
                    }
                    //LOG(INFO) << "pos norm2_i_j is" << norm2_i_j;
                    //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                    if(norm2_i_j > 10.0)
                    {norm2_i_j = 10.0;}
                    ts_local_pos = (std::exp(norm2_i_j) - 1.0) / (std::exp(norm2_i_j) + 1.0);
                    dis_local_pos_data[index_anchor + i_local * height_local + j_local] = ts_local_pos;
                    //dis_local_pos_data[i * height_local * height_local + j_local * height_local + i_local] = ts_local_pos;
                }
            }
		}
        total_dis_local_pos_data[i] = CalcuteShortestPath(dis_local_pos_data, index_anchor, index_local_pos_data, height_local);
        //LOG(INFO) << "total_dis_local_pos_data is" << " " << " "<< total_dis_local_pos_data[i];
	}
}

template <typename Dtype>
Dtype RankHardLossLayer<Dtype>::CalcuteShortestPath(Dtype*local_mat, int index_anchor, Dtype* index_local_data, int height_local)
{
    if(local_mat == NULL)
    {
        LOG(ERROR) << "local distance mat is NULL !!";
        return (-1.0);
    }
    Dtype **S = new Dtype*[height_local];
    for(int i = 0; i < height_local; i++)
    {
        S[i] = new Dtype[height_local];
    }
    S[0][0] = local_mat[index_anchor + 0 * height_local + 0];
    //calculate the first column value
    for(int i = 1; i < height_local; i++)
    {
        S[i][0] = S[i - 1][0] + local_mat[index_anchor + i * height_local + 0];
        index_local_data[index_anchor + i * height_local + 0] = 0;
    }
    //calculate the first row value
    for(int j = 1; j < height_local; j++)
    {
        S[0][j] = S[0][j - 1] + local_mat[index_anchor + 0 * height_local + j];
        index_local_data[index_anchor + 0 * height_local + j] = 1;
    }
    //calculate other value
    for(int i = 1; i < height_local; i++)
    {
        for (int j = 1; j < height_local; j++)
        {
            int direction = S[i][j - 1] < S[i - 1][j] ? 1 : 0;
            S[i][j] = (direction ? S[i][j - 1] : S[i - 1][j]) + local_mat[index_anchor + i * height_local + j];
            index_local_data[index_anchor + i * height_local + j] = direction;
        }
    }

    //LOG(INFO) << "S[height_local - 1][height_local - 1] is" << " " << S[height_local - 1][height_local - 1] ;
    return S[height_local - 1][height_local - 1];


}


template <typename Dtype>
void RankHardLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    //const Dtype* bottom_data = bottom[0]->cpu_data();
    //const Dtype* label = bottom[1]->cpu_data();
    //int count = bottom[0]->count();
	int num = bottom[0]->num();

    int neg_hard_num = this->layer_param_.rank_hard_loss_param().neg_num();
    int pos_hard_num = this->layer_param_.rank_hard_loss_param().pos_num();
    CHECK_EQ(neg_hard_num, pos_hard_num);
	float margin = this->layer_param_.rank_hard_loss_param().margin();
	Dtype* dis_data = dis_.mutable_cpu_data();
    Dtype* total_dis_local_neg_data = total_dis_local_neg_.mutable_cpu_data();
    Dtype* total_dis_local_pos_data = total_dis_local_pos_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();
	set_mask(bottom);
	Dtype loss = 0;
    Dtype loss_local = 0;
    int cnt = neg_hard_num * num;
    for(int i = 0; i < num; i++)
	{
        Dtype pos_dis = 0;
        Dtype neg_dis = 0;
        Dtype pos_dis_local = 0;
        Dtype neg_dis_local = 0;
		for(int j = 0; j < num; j ++)
		{
			if(mask_data[i * num + j] == 0) continue;
            if(mask_data[i * num + j] == 1)
            {
                neg_dis += dis_data[i * num + j];
                neg_dis_local += total_dis_local_neg_data[j];
                //LOG(INFO)<< "total_dis_local_neg_data[j] is" << total_dis_local_neg_data[j];
            }
            if(mask_data[i * num + j] == 2)
            {
                pos_dis += dis_data[i * num + j];
                pos_dis_local += total_dis_local_pos_data[j];
                //LOG(INFO)<< "total_dis_local_pos_data[j] is" << total_dis_local_pos_data[j];
            }
		}
        loss += max(Dtype(0), pos_dis - neg_dis + neg_hard_num * Dtype(margin));
        loss_local += max(Dtype(0), pos_dis_local - neg_dis_local + neg_hard_num * Dtype(margin));

	}

    loss = loss / cnt;
    loss_local = loss_local / cnt;
    //LOG(INFO) << "loss_local and loss is " << loss_local << " " << loss;
    top[0]->mutable_cpu_data()[0] = loss + loss_local;
}

template <typename Dtype>
void RankHardLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {


	const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_data_local = bottom[2]->cpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff_local = bottom[2]->mutable_cpu_diff();
	int count = bottom[0]->count();
    int count_local = bottom[2]->count();
	int num = bottom[0]->num();
	int dim = bottom[0]->count() / bottom[0]->num();
    int dim_local = bottom[2]->count() / bottom[2]->num();
    int channel_local = bottom[2]->channels();
    int height_local = bottom[2]->height();

    int neg_hard_num = this->layer_param_.rank_hard_loss_param().neg_num();
    int pos_hard_num = this->layer_param_.rank_hard_loss_param().pos_num();
    CHECK_EQ(neg_hard_num, pos_hard_num);
	float margin = this->layer_param_.rank_hard_loss_param().margin();

	Dtype* dis_data = dis_.mutable_cpu_data();
    Dtype* total_dis_local_neg_data = total_dis_local_neg_.mutable_cpu_data();
    Dtype* total_dis_local_pos_data = total_dis_local_pos_.mutable_cpu_data();
    Dtype* index_local_neg_data = index_local_neg_.mutable_cpu_data();
    Dtype* index_local_pos_data = index_local_pos_.mutable_cpu_data();
	Dtype* mask_data = mask_.mutable_cpu_data();

	for(int i = 0; i < count; i ++ )
		bottom_diff[i] = 0;
    for(int i = 0; i < count_local; i ++ )
        bottom_diff_local[i] = 0;

    int cnt = neg_hard_num * num;
    for(int i = 0; i < num; i++)
	{
        Dtype pos_dis = 0;
        Dtype neg_dis = 0;
        Dtype pos_dis_local = 0;
        Dtype neg_dis_local = 0;
        const Dtype* fori = bottom_data + i * dim;
        Dtype* fori_diff = bottom_diff + i * dim;

        for(int j = 0; j < num; j ++)
        {
            if(mask_data[i * num + j] == 0) continue;
            if(mask_data[i * num + j] == 1)
            {
                neg_dis += dis_data[i * num + j];
                neg_dis_local += total_dis_local_neg_data[j];
            }
            if(mask_data[i * num + j] == 2)
            {
                pos_dis += dis_data[i * num + j];
                pos_dis_local += total_dis_local_pos_data[j];
            }
        }
        Dtype tloss_tmp = max(Dtype(0), pos_dis - neg_dis + neg_hard_num * Dtype(margin));
        Dtype tloss_tmp_local = max(Dtype(0), pos_dis_local - neg_dis_local + neg_hard_num * Dtype(margin));
        //LOG(INFO) << "tloss_tmp and tloss_tmp_local is " << tloss_tmp << " " << tloss_tmp_local;
        //global gradient
//        if(tloss_tmp == 0)
//        {
//            Dtype pos_dis = 0;
//            Dtype neg_dis = 0;
//            Dtype pos_dis_local = 0;
//            Dtype neg_dis_local = 0;
//            const Dtype* fori = bottom_data + i * dim;
//            Dtype* fori_diff = bottom_diff + i * dim;
//            for(int k = 0; k < dim; k ++)
//            {
//              LOG(INFO) << fori[k];
//              LOG(INFO) << fori_diff[k];
//            }

//            for(int j = 0; j < num; j ++)
//            {
//                if(mask_data[i * num + j] == 0) continue;
//                if(mask_data[i * num + j] == 1)
//                {
//                    neg_dis += dis_data[i * num + j];
//                    neg_dis_local += total_dis_local_neg_data[j];
//                }
//                if(mask_data[i * num + j] == 2)
//                {
//                    pos_dis += dis_data[i * num + j];
//                    pos_dis_local += total_dis_local_pos_data[j];
//                    for(int i = 0; i < num*num; i++)
//                    {
//                        LOG(INFO) << dis_data[i];
//                    }
//                    for(int i = 0; i <num ; i++)
//                    {
//                        LOG(INFO) << total_dis_local_pos_data[i];
//                    }
//                }
//            }
//            Dtype tloss_tmp = max(Dtype(0), pos_dis - neg_dis + neg_hard_num * Dtype(margin));
//            Dtype tloss_tmp_local = max(Dtype(0), pos_dis_local - neg_dis_local + neg_hard_num * Dtype(margin));
//            //LOG(INFO) << "tloss_tmp and tloss_tmp_local is " << tloss_tmp << " " << tloss_tmp_local;
//        }

        if(tloss_tmp > 0)
        {
            for(int j = 0; j < num; j ++)
            {
                if(mask_data[i * num + j] == 1)
                {
                    const Dtype* fneg = bottom_data + j * dim;
                    Dtype* fneg_diff = bottom_diff + j * dim;
                    for(int k = 0; k < dim; k ++)
                    {
                        fori_diff[k] += fneg[k];
                        fneg_diff[k] += fori[k];
                    }
                }
                if(mask_data[i * num + j] == 2)
                {
                    const Dtype* fpos = bottom_data + j * dim;
                    Dtype* fpos_diff = bottom_diff + j * dim;
                    for(int k = 0; k < dim; k ++)
                    {
                        fori_diff[k] += -fpos[k];
                        fpos_diff[k] += -fori[k];
                    }
                }
            }
        }
        //local gradient
        const Dtype* fori_local = bottom_data_local + i * dim_local;
        Dtype* fori_diff_local = bottom_diff_local + i * dim_local;
        if(tloss_tmp_local > 0)
        {
            for(int j = 0; j < num; j ++)
            {
                if(mask_data[i * num + j] == 1)
                {
                    const Dtype* fneg_local = bottom_data_local + j * dim_local;
                    Dtype* fneg_diff_local = bottom_diff_local + j * dim_local;

                    {
                        //[0, 0] are always taken part in diff calculate
                        Dtype norm2_i_j = 0;
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            norm2_i_j += (fori_local[0 * channel_local + k_local] - fneg_local[0 * channel_local + k_local])
                                       * (fori_local[0 * channel_local + k_local] - fneg_local[0 * channel_local + k_local]);
                        }
                        if(norm2_i_j > 10.0)
                        {norm2_i_j = 10.0;}
                        //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                        Dtype exp_norm2 = std::exp(norm2_i_j);
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            fori_diff_local[0 * channel_local + k_local] +=
                                    -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[0 * channel_local + k_local] - fneg_local[0 * channel_local + k_local]);
                            fneg_diff_local[0 * channel_local + k_local] +=
                                    exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[0 * channel_local + k_local] - fneg_local[0 * channel_local + k_local]);
                        }
                        //[6, 6] are always taken part in diff calculate
                        norm2_i_j = 0;
                        int height_local_index = height_local - 1;
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            norm2_i_j += (fori_local[height_local_index * channel_local + k_local] - fneg_local[height_local_index * channel_local + k_local])
                                    * (fori_local[height_local_index * channel_local + k_local] - fneg_local[height_local_index * channel_local + k_local]);
                        }
                        if(norm2_i_j > 10.0)
                        {norm2_i_j = 10.0;}
                        //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                        exp_norm2 = std::exp(norm2_i_j);
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            fori_diff_local[height_local_index * channel_local + k_local] +=
                                    -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[height_local_index * channel_local + k_local] - fneg_local[height_local_index * channel_local + k_local]);
                            fneg_diff_local[height_local_index * channel_local + k_local] +=
                                    exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[0 * channel_local + k_local] - fneg_local[height_local_index * channel_local + k_local]);
                        }
                    }

                    int i_local = height_local - 1;
                    int j_local = height_local - 1;
                    while(i_local > 0 || j_local > 0)
                    {
                        int index_anchor = i * height_local * height_local;
                        if(index_local_neg_data[index_anchor + i_local * height_local + j_local] == 0)
                        {
                            i_local = i_local - 1;
                            j_local = j_local;
                            Dtype norm2_i_j = 0;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                norm2_i_j += (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local])
                                        * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                            }
                            if(norm2_i_j > 10.0)
                            {norm2_i_j = 10.0;}
                            //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                            Dtype exp_norm2 = std::exp(norm2_i_j);
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                fori_diff_local[i_local * channel_local + k_local] +=
                                        -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                                fneg_diff_local[i_local * channel_local + k_local] +=
                                        exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                            }

                        }
                        else if(index_local_neg_data[index_anchor + i_local * height_local + j_local] == 1)
                        {
                            i_local = i_local;
                            j_local = j_local - 1;
                            Dtype norm2_i_j = 0;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                norm2_i_j += (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local])
                                        * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                            }
                            if(norm2_i_j > 10.0)
                            {norm2_i_j = 10.0;}
                            //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                            Dtype exp_norm2 = std::exp(norm2_i_j);
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                fori_diff_local[i_local * channel_local + k_local] +=
                                        -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                                fneg_diff_local[i_local * channel_local + k_local] +=
                                        exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fneg_local[j_local * channel_local + k_local]);
                            }
                        }
                        else
                        {
                            LOG(ERROR) << "direction mat is not correct !!";
                            return;
                        }
                    }
                }
                if(mask_data[i * num + j] == 2)
                {
                    const Dtype* fpos_local = bottom_data_local + j * dim_local;
                    Dtype* fpos_diff_local = bottom_diff_local + j * dim_local;
                    //[0, 0] are always taken part in diff calculate
                    {
                        Dtype norm2_i_j = 0;
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            norm2_i_j += (fori_local[0 * channel_local + k_local] - fpos_local[0 * channel_local + k_local])
                                       * (fori_local[0 * channel_local + k_local] - fpos_local[0 * channel_local + k_local]);
                        }
                        if(norm2_i_j > 10.0)
                        {norm2_i_j = 10.0;}
                        //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                        Dtype exp_norm2 = std::exp(norm2_i_j);
                        //LOG(INFO) << "exp_norm2 is "<< exp_norm2;

                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            fori_diff_local[0 * channel_local + k_local] +=
                                    exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[0 * channel_local + k_local] - fpos_local[0 * channel_local + k_local]);
                            fpos_diff_local[0 * channel_local + k_local] +=
                                    -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[0 * channel_local + k_local] - fpos_local[0 * channel_local + k_local]);
                        }
                        //[6, 6] are always taken part in diff calculate
                        norm2_i_j = 0;
                        int height_local_index = height_local - 1;
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            norm2_i_j += (fori_local[height_local_index * channel_local + k_local] - fpos_local[height_local_index * channel_local + k_local])
                                       * (fori_local[height_local_index * channel_local + k_local] - fpos_local[height_local_index * channel_local + k_local]);
                        }
                        if(norm2_i_j > 10.0)
                        {norm2_i_j = 10.0;}
                        //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                        exp_norm2 = std::exp(norm2_i_j);
                        //LOG(INFO) << "exp_norm2 is "<< exp_norm2;
                        for(int k_local = 0; k_local < channel_local; k_local++)
                        {
                            fori_diff_local[height_local_index * channel_local + k_local] +=
                                    exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[height_local_index * channel_local + k_local] - fpos_local[height_local_index * channel_local + k_local]);
                            fpos_diff_local[height_local_index * channel_local + k_local] +=
                                    -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                    * 2.0 * (fori_local[height_local_index * channel_local + k_local] - fpos_local[height_local_index * channel_local + k_local]);
                        }
                    }

                    int i_local = height_local - 1;
                    int j_local = height_local - 1;
                    while(i_local > 0 || j_local > 0)
                    {
                        int index_anchor = i * height_local * height_local;
                        if(index_local_pos_data[index_anchor + i_local * height_local + j_local] == 0)
                        {
                            i_local = i_local - 1;
                            j_local = j_local;
                            Dtype norm2_i_j = 0;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                norm2_i_j += (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local])
                                           * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                            }
                            if(norm2_i_j > 10.0)
                            {norm2_i_j = 10.0;}
                            //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                            Dtype exp_norm2 = std::exp(norm2_i_j);
                            //LOG(INFO) << "norm2_i_j is "<< norm2_i_j;
                            //LOG(INFO) << "exp_norm2 is "<< exp_norm2;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                fori_diff_local[i_local * channel_local + k_local] +=
                                        exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                                fpos_diff_local[i_local * channel_local + k_local] +=
                                        -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                            }

                        }
                        else if(index_local_pos_data[index_anchor + i_local * height_local + j_local] == 1)
                        {
                            i_local = i_local;
                            j_local = j_local - 1;
                            Dtype norm2_i_j = 0;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                norm2_i_j += (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local])
                                           * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                            }
                            if(norm2_i_j > 10.0)
                            {norm2_i_j = 10.0;}
                            //norm2_i_j = std::sqrt(norm2_i_j); comment this line for easy gradient calculate
                            Dtype exp_norm2 = std::exp(norm2_i_j);
                            //LOG(INFO) << "exp_norm2 is "<< exp_norm2;
                            for(int k_local = 0; k_local < channel_local; k_local++)
                            {
                                fori_diff_local[i_local * channel_local + k_local] +=
                                        exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                                fpos_diff_local[i_local * channel_local + k_local] +=
                                        -exp_norm2 * (1.0 / (exp_norm2 + 1.0) / (exp_norm2 + 1.0))
                                        * 2.0 * (fori_local[i_local * channel_local + k_local] - fpos_local[j_local * channel_local + k_local]);
                            }
                        }
                        else
                        {
                            LOG(ERROR) << "direction mat is not correct !!";
                            return;
                        }
                    }
                }
            }
        }

    }

	for (int i = 0; i < count; i ++)
	{
        bottom_diff[i] = bottom_diff[i] / cnt;
	}
    for (int i = 0; i < count_local; i ++)
    {
        bottom_diff_local[i] = bottom_diff_local[i] / cnt;
    }

}

#ifdef CPU_ONLY
STUB_GPU(RankHardLossLayer);
#endif

INSTANTIATE_CLASS(RankHardLossLayer);
REGISTER_LAYER_CLASS(RankHardLoss);


}   // namespace caffe
