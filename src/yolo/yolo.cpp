#include <iostream>   
#include <stdio.h>

#include <string>
#include <vector>
#include <math.h>
#include <functional>                                   // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                       // 线程
#include <queue>                                        // 队列
#include <mutex>                                        // 线程锁
#include <chrono>                                       // 时间库
#include <memory>                                       // 智能指针
#include <future>                                       // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                           // 线程通信库
#include <filesystem> 
#include <unistd.h>
#include <dirent.h>                                     // opendir和readdir包含在这里
#include <sys/stat.h>
#include <map>
#include <fstream>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "spdlog/sinks/basic_file_sink.h"               // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "yolo/yolo.h"
#include "yolo/model-utils.h"


using namespace std;
using namespace cv;
using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;};

const int anchors[3][6] = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};                  

struct Job{
    shared_ptr<promise<vector<Result>>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    vector<Mat> input_images;                       // 输入图像, 多batch 
    vector<int> heights;
    vector<int> widths; 
    string channel_id;
    vector<long> timestamps;
    vector<unsigned char*> input_images_data; 
    vector<string> unique_ids;
    bool inferLog{false};                                  // 是否打印日志
};

class InferImpl : public InferInterface{                                        
public:
    virtual ~InferImpl(){
        stop();
        spdlog::warn("Destruct instance done!");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
            products_cv_.notify_one();                                          // 通知boss给break掉
        }
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(
        const string& file, 
        const size_t& md,
        string& device,
        bool modelLog=false,
        bool multi_label=true
    ){
        
        multi_label_ = multi_label;
        model_path_ = file;
        max_det = md;
        modelLog_ = modelLog;
        running_ = true;                                                        // 启动后，运行状态设置为true
        device_ = device;
        string modelName = getFilename(model_path_, true);
        vector<string> splits = splitString(modelName, "-");
        if (splits[0] == "v8" || splits[0] == "v11")
            is_v5_ = false;
        promise<bool> pro;
        if (modelLog_){
            print_avaliable_devices();
            spdlog::info("Using device {}", device_);
        }
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    // --------------------------------------------------
    bool startup(
        string nickName,
        PushResult callback,
        void* userP,
        const string& file, 
        const size_t& md,
        int& mq,
        string& device,
        bool modelLog=false,
        bool multi_label=true
    ){
        nick_name_ = nickName;
        callback_ = callback;
        userP_ = userP;
        multi_label_ = multi_label;
        // is_qsize_set_和max_qsize_为静态成员变量
        if (!is_qsize_set_){
            max_qsize_ = mq;
            is_qsize_set_ = true;
        }
        model_path_ = file;
        max_det = md;
        modelLog_ = modelLog;
        running_ = true;                                                        // 启动后，运行状态设置为true
        device_ = device;
        promise<bool> pro;
        string modelName = getFilename(model_path_, true);
        vector<string> splits = splitString(modelName, "-");
        if (splits[0] == "v8" || splits[0] == "v11")
            is_v5_ = false;
        if (modelLog_){
            print_avaliable_devices();
            spdlog::info("Using device {}", device_);
        }
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }
    // --------------------------------------------

    void worker(promise<bool>& pro){
        // 加载模型
        ov::Core core;
        ov::CompiledModel model = core.compile_model(model_path_, device_);
        request = model.create_infer_request();
        if(model.inputs().empty() || model.outputs().empty()){
            // failed
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            spdlog::error("Load model failed from path: {}!", model_path_);
            return;
        }
        string old_suffix = "xml"; string new_suffix = "txt";
        string model_classes_path = replaceSuffix(model_path_, old_suffix, new_suffix);
        CURRENT_IDX2CLS = readFileToMap(model_classes_path);
        // load success
        pro.set_value(true);  
        if (modelLog_) spdlog::info("Model loaded successfully from {}", model_path_);

        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                Job job_one = std::move(jobs_.front());
                jobs_.pop();                                                    // 从jobs_任务队列中将当前要推理的job给pop出来 
                l.unlock();                                                     // 注意这里要解锁, 否则调用inference等inference执行完再解锁又变同步了
                inference(job_one);                                             // 调用inference执行推理
            }
        }
    }

    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, float conf_thre, float nms_thre, bool inferLog=false) override{  
        Job job;
        vector<string> unique_ids;      

        for (int i = 0; i < n_images; ++i){
            job.heights.push_back(inputs[i].height);
            job.widths.push_back(inputs[i].width);
            int numel = inputs[i].height * inputs[i].width * 3;
            unique_ids.push_back(inputs[i].unique_id);
            cv::Mat image_one(inputs[i].height, inputs[i].width, CV_8UC3);
            memcpy(image_one.data, inputs[i].data, numel);
            // string save_path = "images-received/" + inputs[i].unique_id + ".jpg";
            // cv::cvtColor(image_one, image_one, cv::COLOR_RGB2BGR);
            // cv::imwrite(save_path, image_one);
            job.input_images.push_back(image_one);
            job.input_images_data.push_back(inputs[i].data);
            job.timestamps.push_back(inputs[i].timestamp);
        }            

        conf_thre_ = conf_thre; 
        nms_thre_ = nms_thre;    
        job.pro.reset(new promise<vector<Result>>());
        job.unique_ids = unique_ids;
        job.inferLog = inferLog;

        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            unique_lock<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    // --------------------------------------------
    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    std::shared_future<std::vector<Result>> forward(Job& job){             
        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            unique_lock<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }
    // --------------------------------------------

    void preprocess(
        vector<Mat>& batched_imgs, 
        ov::InferRequest& request, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors, 
        int& curr_batch_size, 
        size_t& max_det
    ){
        // set input & ouput shape for dynamic batch 
        ov::Tensor input_tensor = request.get_input_tensor();
        ov::Shape input_shape = input_tensor.get_shape();
        input_shape[0] = curr_batch_size; // Set the batch size in the input shape
        input_tensor.set_shape(input_shape);
        size_t input_channel = input_shape[1];
        size_t input_height = input_shape[2];
        size_t input_width = input_shape[3];

        ov::Tensor output_tensor = request.get_output_tensor(0);
        ov::Shape output_shape = output_tensor.get_shape();
        auto output_shape_size = output_shape.size();
        if (output_shape_size == 2) {
            output_shape[0] = curr_batch_size * max_det;     // 如果输出维度是2维的话, 说明已经将NMS添加到网络中了, 输出维度应该为[0, 7], 7代表 xyxy + conf + cls_id + image_idx
        } else {
            output_shape[0] = curr_batch_size;               // 如果输出维度为3维的话, 说明NMS未添加到网络中, 输出维度应该为[0, 15120, nc+5]
        }
        output_tensor.set_shape(output_shape);

        if (inferLog_) {
            spdlog::info("Model input shape: {} x {} x {} x {}", curr_batch_size, input_channel, input_height, input_width);
            if (output_shape_size == 2){
                spdlog::info("Model max output shape: {} x {}", output_shape[0], output_shape[1]);
            } else if (output_shape_size == 3) {
                spdlog::info("Model max output shape: {} x {} x {}", output_shape[0], output_shape[1], output_shape[2]);
            } else {
                spdlog::info("Model max output shape: {} x {} x {} x {}", output_shape[0], output_shape[1], output_shape[2],  output_shape[3]);
            }
        }

        // reize and pad
        for (int i = 0; i < batched_imgs.size(); ++i){
            Mat& img = batched_imgs[i];
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();

            float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
                            static_cast<float>(input_height) / static_cast<float>(img.rows));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
            cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);
        }

        // HWC-->CHW & /255. & transfer data to input_tensor
        // ! 注意, 给我的图片是RGB的, 所以这里没有BGR转到RGB的过程
        float* input_data_host = input_tensor.data<float>();
        float* i_input_data_host;
        size_t img_area = input_height * input_width;
        for (int i = 0; i < batched_imgs.size(); ++i){
            i_input_data_host = input_data_host + img_area * 3 * i;
            unsigned char* pimage = batched_imgs[i].data;
            float* phost_r = i_input_data_host + img_area * 0;
            float* phost_g = i_input_data_host + img_area * 1;
            float* phost_b = i_input_data_host + img_area * 2;
            for(int j = 0; j < img_area; ++j, pimage += 3){
                *phost_r++ = pimage[0] / 255.0f ;
                *phost_g++ = pimage[1] / 255.0f;
                *phost_b++ = pimage[2] / 255.0f;
            }
        }
    }

    void do_infer(ov::InferRequest & request){
        request.infer();
    }

    void clip_boxes(
        float& box_left, 
        float& box_right, 
        float& box_top, 
        float& box_bottom, 
        vector<int>& img_org_shape
    ){
        auto clip_value = [](float value, float min_value, float max_value) {
            return (value < min_value) ? min_value : (value > max_value) ? max_value : value;
        };
        int org_height = img_org_shape[0];
        int org_width = img_org_shape[1];
        box_left = clip_value(box_left, 0, org_width);
        box_right = clip_value(box_right, 0, org_width);
        box_top = clip_value(box_top, 0, org_height);
        box_bottom = clip_value(box_bottom, 0, org_height);
    }

    void postprocess(
        ov::InferRequest & request,
        int curr_batch_size,
        vector<Result>& results, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<vector<int>>& batched_imgs_org_shape
    ){
        ov::Tensor output_tensor = request.get_output_tensor(0);
        ov::Shape output_shape = output_tensor.get_shape();
        int output_shape_size = output_shape.size();

        for (int i_img = 0; i_img < curr_batch_size; i_img++){
            vector<Box> bboxes;       // 初始化变量bboxes:[[x1, y1, x2, y2, conf, label], [x1, y1, x2, y2, conf, label]...]
            if (is_v5_){
                if (output_shape_size == 3){
                    decode_boxes_1output_v5(i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);
                } else {
                    decode_boxes_3output_v5(i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);
                }
            }else{
                decode_boxes_1output_v8_v11_detect(i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);
            }

            if (inferLog_) spdlog::info("Decoded bboxes.size = {}", bboxes.size());

            // nms非极大抑制
            // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
            std::sort(bboxes.begin(), bboxes.end(), [](Box &a, Box &b)
                    { return a.score > b.score; });
            std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
            // 定义一个lambda的iou函数
            auto iou = [](const Box &a, const Box &b)
            {
                float cross_left = std::max(a.left, b.left);
                float cross_top = std::max(a.top, b.top);
                float cross_right = std::min(a.right, b.right);
                float cross_bottom = std::min(a.bottom, b.bottom);

                float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
                float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
                if (cross_area == 0 || union_area == 0)
                    return 0.0f;
                return cross_area / union_area;
            };

            for (int i = 0; i < bboxes.size(); ++i){
                if (remove_flags[i])
                    continue;                                       // 如果已经被标记为需要移除，则continue

                auto &ibox = bboxes[i];                             // 获得第i个box
                auto float2int = [] (float x) {return static_cast<int>(round(x));};
                // vector<int> _ibox = {float2int(ibox[0]), float2int(ibox[1]), float2int(ibox[2]), float2int(ibox[3])};
                // results[i_img].boxes.emplace_back(_ibox);           // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
                // results[i_img].labels.emplace_back(int(ibox[4]));
                // results[i_img].scores.emplace_back(ibox[5]);
                Box bbox;
                bbox.left = float2int(ibox.left); 
                bbox.top = float2int(ibox.top);
                bbox.right = float2int(ibox.right);
                bbox.bottom = float2int(ibox.bottom);
                bbox.label = int(ibox.label);
                bbox.score = ibox.score;
                bbox.keypoints = ibox.keypoints;
                results[i_img].bboxes.emplace_back(bbox);
                for (int j = i + 1; j < bboxes.size(); ++j){        // 遍历剩余框，与box_result中的框做iou
                    if (remove_flags[j])
                        continue;                                   // 如果已经被标记为需要移除，则continue

                    auto &jbox = bboxes[j];                         // 获得第j个box
                    if (ibox.label == jbox.label){ 
                        // class matched
                        if (iou(ibox, jbox) >= nms_thre_)       // iou值大于阈值，将该框标记为需要remove
                            remove_flags[j] = true;
                    }
                }
            }
            if (inferLog_) spdlog::info("box_result.size = {}", results[i_img].bboxes.size());
        }
    }

    void decode_boxes_1output_v5(
        int& i_img, 
        vector<Box>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
        ){
        ov::Tensor output = request.get_output_tensor(0);
        size_t output_numbox = output.get_shape()[1];
        size_t output_numprob = output.get_shape()[2];
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        float* output_data_host = (float*)output.data();                                    // fetch index 0 because there is only one output   
        int num_classes = output_numprob - 5;
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        for (int i = 0; i < output_numbox; ++i) {
            Box box_one;
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob
            float objness = ptr[4];                                                         // 获得置信度
            if (objness < conf_thre_)
                continue;

            float *pclass = ptr + 5;                                                        // 获得类别开始的地址
            label = max_element(pclass, pclass + num_classes) - pclass;                     // 获得概率最大的类别
            prob = pclass[label];                                                           // 获得类别概率最大的概率值
            confidence = prob * objness;                                                    // 计算后验概率
            if (confidence < conf_thre_)
                continue;
            if (multi_label_)
                while (confidence >= conf_thre_){
                    labels.push_back(label);
                    confidences.push_back(confidence);
                    *(pclass + label) = 0.;
                    label = max_element(pclass, pclass + num_classes) - pclass;
                    prob = pclass[label];
                    confidence = prob * objness;
                }                   

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                             // x1
            float image_base_right = (right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                           // x2
            float image_base_top = (top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                               // y1
            float image_base_bottom = (bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];                         // y2
            clip_boxes(image_base_left, image_base_right, image_base_top, image_base_bottom, img_org_shape);
            box_one.left = image_base_left;
            box_one.top = image_base_top;
            box_one.right = image_base_right;
            box_one.bottom = image_base_bottom;
            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j){
                    box_one.label = (float)labels[j];
                    box_one.score = confidences[j];
                    bboxes.push_back(box_one);
                }
                labels.clear();
                confidences.clear();
            }else{
                box_one.label = (float)label;
                box_one.score = confidence;
                bboxes.push_back(box_one);  // 放进bboxes中
            }
        }
    }
    
    void decode_boxes_3output_v5(
        int& i_img,
        vector<Box>& bboxes,
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        int anchor_size = sizeof(anchors[0]) / sizeof(int) / 2;
        vector<ov::Tensor> outputs;
        vector<ov::Shape> output_shapes;
        vector<size_t> offsets;
        for (int i = 0; i < 3; ++i) {
            ov::Tensor i_output = request.get_output_tensor(i);
            ov::Shape i_output_shape = i_output.get_shape();
            size_t i_offset =  i_output_shape[1] * i_output_shape[2] * i_output_shape[3];
            outputs.push_back(i_output);
            output_shapes.push_back(i_output_shape);
            offsets.push_back(i_offset);
        }

        // decode boxes
        int prob_box_size = output_shapes[0][1] / anchor_size;
        int nc = prob_box_size - 5;
        multi_label_ = multi_label_ && (nc > 1);
        int label; float prob; float confidence; float iprob;
        vector<int> labels; vector<float> confidences;
        // iter every output
        for (int ifeat = 0; ifeat < 3; ++ifeat) {
            int grid_h = output_shapes[ifeat][2];
            int grid_w = output_shapes[ifeat][3];
            int grid_len = grid_w * grid_h;
            int stride = request.get_input_tensor().get_shape()[3] / grid_w;
            float* output_data_host = (float*)outputs[ifeat].data() + i_img * offsets[ifeat];
            // iter every anchor
            for (int a = 0; a < anchor_size; a++){
                // iter every h grid                        
                for (int i = 0; i < grid_h; i++){
                    // iter every w grid
                    for (int j = 0; j < grid_w; j++){
                        // 想象成一个魔方, j是w维度, i是h维度, a是深度
                        Box box_one;
                        // xyxy + conf + cls_prob
                        float objness = output_data_host[(prob_box_size * a + 4) * grid_len + i * grid_w + j];
                        if (objness > conf_thre_){
                            int offset = (prob_box_size * a) * grid_len + i * grid_w + j;
                            float *in_ptr = output_data_host + offset;
                            float box_x = in_ptr[0 * grid_len] * 2.0 - 0.5;
                            float box_y = in_ptr[1 * grid_len] * 2.0 - 0.5;
                            float box_w = in_ptr[2 * grid_len] * 2.0;
                            float box_h = in_ptr[3 * grid_len] * 2.0;
                            box_x = (box_x + j) * (float)stride;
                            box_y = (box_y + i) * (float)stride;
                            box_w = box_w * box_w * (float)anchors[ifeat][a * 2];
                            box_h = box_h * box_h * (float)anchors[ifeat][a * 2 + 1];
                            float box_x1 = box_x - (box_w / 2.0);
                            float box_y1 = box_y - (box_h / 2.0);
                            float box_x2 = box_x + (box_w / 2.0);
                            float box_y2 = box_y + (box_h / 2.0);
                            prob = 0;
                            for (int t = 0; t < nc; ++t){
                                iprob = in_ptr[(5 + t) * grid_len];
                                if (iprob > prob){
                                    label = t;
                                    prob = iprob;
                                }
                            }
                            confidence = prob * objness;                                          // 计算后验概率
                            if (confidence < conf_thre_)
                                continue;
                            if (multi_label_){
                                while (confidence >= conf_thre_){
                                    labels.push_back(label);
                                    confidences.push_back(confidence);
                                    in_ptr[(5 + label) * grid_len] = 0.;
                                    prob = 0;
                                    for (int t = 0; t < nc; ++t){
                                        iprob = in_ptr[(5 + t) * grid_len];
                                        if (iprob > prob){
                                            label = t;
                                            prob = iprob;
                                        }
                                    }
                                    confidence = prob * objness;
                                } 
                            }

                            float image_base_left = (box_x1 - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                                                                     // x1
                            float image_base_right = (box_x2 - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                                                                   // x2
                            float image_base_top = (box_y1 - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                                                                       // y1
                            float image_base_bottom = (box_y2 - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];
                            clip_boxes(image_base_left, image_base_right, image_base_top, image_base_bottom, img_org_shape);
                            box_one.left = image_base_left;
                            box_one.top = image_base_top;
                            box_one.right = image_base_right;
                            box_one.bottom = image_base_bottom;
                            if (multi_label_){
                                for (int j = 0; j < labels.size(); ++j){
                                    box_one.label = (float)labels[j];
                                    box_one.score = confidences[j];
                                    bboxes.push_back(box_one);
                                }
                                labels.clear();
                                confidences.clear();
                            }else{
                                box_one.label = (float)label;
                                box_one.score = confidence;
                                bboxes.push_back(box_one);  // 放进bboxes中
                            }
                        }
                    }
                }
            }
        }
    }

    void decode_boxes_1output_v8_v11_detect(
        int& i_img, 
        vector<Box>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        ov::Tensor output = request.get_output_tensor(0);
        size_t output_numbox = output.get_shape()[1];
        size_t output_numprob = output.get_shape()[2];
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        float* output_data_host = (float*)output.data();                                    // fetch index 0 because there is only one output   
        int num_classes = output_numprob - 4;
        // decode and filter boxes by conf_thre
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        for (int i = 0; i < output_numbox; ++i) {
            Box box_one;
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob

            float *pclass = ptr + 4;                                                        // 获得类别开始的地址
            label = max_element(pclass, pclass + num_classes) - pclass;                     // 获得概率最大的类别
            prob = pclass[label];                                                           // 获得类别概率最大的概率值
            confidence = prob;                                                    // 计算后验概率
            if (confidence < conf_thre_)
                continue;
            if (multi_label_)
                while (confidence >= conf_thre_){
                    labels.push_back(label);
                    confidences.push_back(confidence);
                    *(pclass + label) = 0.;
                    label = max_element(pclass, pclass + num_classes) - pclass;
                    prob = pclass[label];
                    confidence = prob;
                }                   

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                             // x1
            float image_base_right = (right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                           // x2
            float image_base_top = (top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                               // y1
            float image_base_bottom = (bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];                         // y2
            clip_boxes(image_base_left, image_base_right, image_base_top, image_base_bottom, img_org_shape);
            box_one.left = image_base_left;
            box_one.top = image_base_top;
            box_one.right = image_base_right;
            box_one.bottom = image_base_bottom;
            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j){
                    box_one.label = (float)labels[j];
                    box_one.score = confidences[j];
                    bboxes.push_back(box_one);
                }
                labels.clear();
                confidences.clear();
            }else{
                box_one.label = (float)label;
                box_one.score = confidence;
                bboxes.push_back(box_one);  // 放进bboxes中
            }
        }
    }

    void inference(Job& job){
        // if (InferImpl::warmuped_)
        //     this_thread::sleep_for(chrono::seconds(10));
        int curr_batch_size = job.input_images.size();
        Result dummy; dummy.unique_id = "dummy"; dummy.nick_name = nick_name_;
        std::vector<Result> results(curr_batch_size, dummy);

        vector<Mat> batched_imgs;
        for (int i = 0; i < job.input_images.size(); ++i){
            batched_imgs.push_back(job.input_images[i]);
            results[i].data = job.input_images_data[i];
        }

        inferLog_ = job.inferLog;
        vector<vector<int>> batched_imgs_org_shape;
        for (int i = 0; i < curr_batch_size; ++i){
            int height = batched_imgs[i].rows;
            int width = batched_imgs[i].cols;
            vector<int> i_shape = {height, width};
            batched_imgs_org_shape.push_back(i_shape);
            results[i].unique_id = job.unique_ids[i];           // attach each id to results
            results[i].height = height;
            results[i].width =  width;
            results[i].channel_id = job.channel_id;
            results[i].timestamp = job.timestamps[i];
        }

        #ifdef WITH_CLOCKING
            auto tiktok = time_point::now();
        #endif
            // preprocess 
            vector<vector<int>> batched_pad_w, batched_pad_h;
            vector<float> batched_scale_factors;
            preprocess(batched_imgs, request, batched_pad_w, batched_pad_h, batched_scale_factors, curr_batch_size, max_det);
        #ifdef WITH_CLOCKING
            InferImpl::records[0].push_back(micros_cast(time_point::now() - tiktok));
            tiktok = time_point::now();
        #endif

            // infer
            do_infer(request);
        #ifdef WITH_CLOCKING
            InferImpl::records[1].push_back(micros_cast(time_point::now() - tiktok));
            tiktok = time_point::now();
        #endif
        
            // postprocess
            postprocess(request, curr_batch_size, results, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape); 
        #ifdef WITH_CLOCKING
            InferImpl::records[2].push_back(micros_cast(time_point::now() - tiktok));
            tiktok = time_point::now();
        #endif

        // return results to future
        job.pro->set_value(results);

        // judge warmuped or not 
        if (!InferImpl::warmuped_){
            InferImpl::warmuped_times_++;
            if (InferImpl::warmuped_times_ <= InferImpl::total_warmup_times_)
                return;
        }
        InferImpl::warmuped_ = true;
        
        // callback
        if (!(callback_ == nullptr)){
            reIndexResults(results, CURRENT_IDX2CLS, UNIFIED_CLS2IDX, nick_name_);
            callback_(results, userP_);
        }

        // // release the image data
        // for (int i = 0; i < batched_imgs.size(); ++i) {
        //     if (!batched_imgs[i].empty()){
        //         batched_imgs[i].release();
        //         delete [] batched_imgs[i].data;
        //     }
        // }
    }
    
    //------------------------------------------------------------------------------------------------------------------
    virtual void warmup(){
        for (int i = 0; i < InferImpl::total_warmup_times_; ++i){
            int height = 1080;
            int width = 1920;
            cv::Mat dummyImage = cv::Mat::zeros(height, width, CV_8UC3);
            int n_images = 1;
            float dummy_conf_thre = 0.1;
            string dummy_channel_id = "ch01";
            int numel = height * width * 3;
            Input dummy_input;
            dummy_input.height = height;
            dummy_input.width = width;
            dummy_input.data = new unsigned char[numel];
            memcpy(dummy_input.data, dummyImage.data, numel);
            Input dummy_inputs[n_images];
            dummy_inputs[0] = dummy_input;
            auto r = this->forward(dummy_inputs, n_images, 0.4, false).get();
            delete [] dummy_input.data;          
        }
    }
    
    virtual bool add_images(Input* inputs, int& n_images, float conf_thre, float nms_thre, string channel_id) override{
        // if exceed max queue length, then pop the front job of the queue and delete the image data 
        // finnally push the new one into the queue
        bool overflow = false;
        if (InferImpl::warmuped_){              // 只有已经热身过了才会去判断是否超过队列长度
            received_++;
            {
                unique_lock<mutex> l(lock_);
                if (jobs_.size() >= max_qsize_){
                    overflow = true;
                    int n_images_of_qfront = jobs_.front().input_images_data.size();
                    for (int i = 0; i < n_images_of_qfront; ++i){
                        unsigned char* image_data_of_qfront = jobs_.front().input_images_data[i];
                        delete [] image_data_of_qfront;
                        throwed_++;
                    }
                    jobs_.pop();
                    double throwed_ratio = (double)throwed_ / (double(received_) + 1e-5);
                    spdlog::debug("Images receieved: {}, throwed: {}, throwed ratio: {}", received_, throwed_, throwed_ratio);
                }
            }
        }

        Job job;
        job.channel_id = channel_id;                             
        for (int i = 0; i < n_images; ++i){
            job.timestamps.push_back(inputs[i].timestamp);
            job.heights.push_back(inputs[i].height);
            job.widths.push_back(inputs[i].width);
            int numel = inputs[i].height * inputs[i].width * 3;
            job.unique_ids.push_back(inputs[i].unique_id);
            cv::Mat image_one(inputs[i].height, inputs[i].width, CV_8UC3);
            memcpy(image_one.data, inputs[i].data, numel);
            job.input_images.push_back(image_one);
            job.input_images_data.push_back(inputs[i].data);
        } 

        conf_thre_ = conf_thre;  
        nms_thre_ = nms_thre;    
        job.pro.reset(new promise<vector<Result>>());

        forward(job);
        return overflow;
    }

    virtual int get_qsize() override{
        int size;
        {
            unique_lock<mutex> l(lock_);
            size = jobs_.size();
        }
        return size;
    }
    //------------------------------------------------------------------------------------------------------------------
    virtual vector<vector<float>> get_records() override{       // 计时相关, 可删
        return InferImpl::records;
    }

private:
    // 可调数据
    string nick_name_;                                         // 模型名称, 为了多模型调度时区分模型用
    string model_path_;                                         // 模型路径
    size_t max_det;
    string device_;
    bool multi_label_;
    bool is_v5_{true};
    float conf_thre_{0.5};
    float nms_thre_{0.4};
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    // 模型初始化有关           
    ov::InferRequest request;
    //日志相关
    bool modelLog_;                                             // 模型加载时是否打印日志在控制台
    bool inferLog_;                                             // 模型推理时是否打印日志在控制台
    std::shared_ptr<spdlog::logger> logger_;                    // logger_负责日志文件, 记录一些错误日志
    string logs_dir{"infer-logs"};                              // 日志文件存放的文件夹   
    // 计时相关
    static vector<vector<float>> records;                       // 计时相关: 静态成员变量声明
    //---------------------------------
    // 事件相关
    PushResult callback_;
    void* userP_;
    static int max_qsize_;                                     
    static bool is_qsize_set_;                                  // 队列长度是否已被设置
    static bool warmuped_;                                      // 是否warmup了
    static int total_warmup_times_;                             // 总共需要warmup几次
    static int warmuped_times_;                                 // 当前warmup了几次
    thread boss_thread_;
    mutex products_lock_;
    condition_variable products_cv_;
    std::map<int, string> CURRENT_IDX2CLS;                      // 当前模型class index到class name的映射: 为了保证同一个类别对应多个模型推理出的结果的index保持一致, 设置该变量, 存储index到class name的映射
    long received_{0};                                             // 统计当前接收了多少
    long throwed_{0};                                              // 统计当前扔掉了多少
    //---------------------------------
};

// 在类体外初始化这几个静态变量
bool InferImpl::is_qsize_set_ = false;                          
int InferImpl::max_qsize_ = 1000;
bool InferImpl::warmuped_ = false;
int InferImpl::total_warmup_times_ = 2;
int InferImpl::warmuped_times_ = 0;
vector<vector<float>> InferImpl::records(3);                    // 计时相关: 静态成员变量定义, 长度为3

shared_ptr<InferInterface> create_infer(std::string &file, int max_det, std::string& device, bool modelLog, bool multi_label){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(file, max_det, device, modelLog, multi_label)){
        instance.reset();                                                     
    }
    return instance;
};

// --------------------------------------------
shared_ptr<InferInterface> create_infer(std::string nickName, PushResult callback, void *userP, std::string &file, int max_det, int max_qsize, std::string& device, bool modelLog, bool multi_label){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(nickName, callback, userP, file, max_det, max_qsize, device, modelLog, multi_label)){
        instance.reset();   
        return nullptr;                                                     
    }
    return instance;
};
// --------------------------------------------