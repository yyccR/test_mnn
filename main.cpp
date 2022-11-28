
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <MNN/Interpreter.hpp>

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

typedef std::vector<int> YoloXAnchor;

const int coco_color_list[80][3] =
        {
                //{255 ,255 ,255}, //bg
                {170 ,  0 ,255},
                { 84 , 84 ,  0},
                { 84 ,170 ,  0},
                { 84 ,255 ,  0},
                {170 , 84 ,  0},
                {170 ,170 ,  0},
                {118 ,171 , 47},
                {238 , 19 , 46},
                {216 , 82 , 24},
                {236 ,176 , 31},
                {125 , 46 ,141},
                { 76 ,189 ,237},
                { 76 , 76 , 76},
                {153 ,153 ,153},
                {255 ,  0 ,  0},
                {255 ,127 ,  0},
                {190 ,190 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 ,255},

                {170 ,255 ,  0},
                {255 , 84 ,  0},
                {255 ,170 ,  0},
                {255 ,255 ,  0},
                {  0 , 84 ,127},
                {  0 ,170 ,127},
                {  0 ,255 ,127},
                { 84 ,  0 ,127},
                { 84 , 84 ,127},
                { 84 ,170 ,127},
                { 84 ,255 ,127},
                {170 ,  0 ,127},
                {170 , 84 ,127},
                {170 ,170 ,127},
                {170 ,255 ,127},
                {255 ,  0 ,127},
                {255 , 84 ,127},
                {255 ,170 ,127},
                {255 ,255 ,127},
                {  0 , 84 ,255},
                {  0 ,170 ,255},
                {  0 ,255 ,255},
                { 84 ,  0 ,255},
                { 84 , 84 ,255},
                { 84 ,170 ,255},
                { 84 ,255 ,255},
                {170 ,  0 ,255},
                {170 , 84 ,255},
                {170 ,170 ,255},
                {170 ,255 ,255},
                {255 ,  0 ,255},
                {255 , 84 ,255},
                {255 ,170 ,255},
                { 42 ,  0 ,  0},
                { 84 ,  0 ,  0},
                {127 ,  0 ,  0},
                {170 ,  0 ,  0},
                {212 ,  0 ,  0},
                {255 ,  0 ,  0},
                {  0 , 42 ,  0},
                {  0 , 84 ,  0},
                {  0 ,127 ,  0},
                {  0 ,170 ,  0},
                {  0 ,212 ,  0},
                {  0 ,255 ,  0},
                {  0 ,  0 , 42},
                {  0 ,  0 , 84},
                {  0 ,  0 ,127},
                {  0 ,  0 ,170},
                {  0 ,  0 ,212},
                {  0 ,  0 ,255},
                {  0 ,  0 ,  0},
                { 36 , 36 , 36},
                { 72 , 72 , 72},
                {109 ,109 ,109},
                {145 ,145 ,145},
                {182 ,182 ,182},
                {218 ,218 ,218},
                {  0 ,113 ,188},
                { 80 ,182 ,188},
                {127 ,127 ,  0},
        };

void nms(std::vector<BoxInfo>& input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1)
                   * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i) {
        for (int j = i + 1; j < int(input_boxes.size());) {
            float xx1 = (std::max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (std::max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (std::min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (std::min)(input_boxes[i].y2, input_boxes[j].y2);
            float w = (std::max)(float(0), xx2 - xx1 + 1);
            float h = (std::max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            if(inter > 0){
                float ovr = inter / (vArea[i] + vArea[j] - inter);
                if (ovr >= NMS_THRESH) {
                    input_boxes.erase(input_boxes.begin() + j);
                    vArea.erase(vArea.begin() + j);
                }
                else {
                    j++;
                }
            }else{
                j++;
            }

        }
    }
}

void generate_anchors(const int target_height, const int target_width, std::vector<int> &strides, std::vector<YoloXAnchor> &anchors)
{
    for (auto stride : strides)
    {
        int num_grid_w = target_width / stride;
        int num_grid_h = target_height / stride;
        for (int g1 = 0; g1 < num_grid_h; ++g1)
        {
            for (int g0 = 0; g0 < num_grid_w; ++g0)
            {
                anchors.push_back((YoloXAnchor) {g0, g1, stride});
            }
        }
    }
}

void draw_coco_bboxes(const cv::Mat& bgr, const std::vector<BoxInfo>& bboxes)
{
    static const char* class_names[] = { "person", "bicycle", "car", "motorcycle", "airplane", "bus",
                                         "train", "truck", "boat", "traffic light", "fire hydrant",
                                         "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                                         "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
                                         "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
                                         "skis", "snowboard", "sports ball", "kite", "baseball bat",
                                         "baseball glove", "skateboard", "surfboard", "tennis racket",
                                         "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                                         "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
                                         "hot dog", "pizza", "donut", "cake", "chair", "couch",
                                         "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
                                         "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
                                         "toaster", "sink", "refrigerator", "book", "clock", "vase",
                                         "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    cv::Mat image = bgr;
    int src_w = image.cols;
    int src_h = image.rows;

    for (size_t i = 0; i < bboxes.size(); i++)
    {
        const BoxInfo& bbox = bboxes[i];
        cv::Scalar color = cv::Scalar(coco_color_list[bbox.label][0],
                                      coco_color_list[bbox.label][1],
                                      coco_color_list[bbox.label][2]);
        //fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f %.2f\n", bbox.label, bbox.score,
        //    bbox.x1, bbox.y1, bbox.x2, bbox.y2);

        cv::rectangle(image, cv::Point(bbox.x1, bbox.y1), cv::Point(bbox.x2, bbox.y2), color);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[bbox.label], bbox.score * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = bbox.x1;
        int y = bbox.y1 - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      color, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255));
    }

    cv::imshow("image", image);
}


int main() {

    std::string image_file = "/Users/yang/CLionProjects/test_mnn/data/images/img.jpg";
    std::string model_file = "/Users/yang/CLionProjects/test_mnn/yolox/yolox_tiny.mnn";

    auto mnnNet = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    MNN::ScheduleConfig netConfig;
    netConfig.type      = MNN_FORWARD_CPU;
    netConfig.numThread = 4;
    MNN::Session* session = mnnNet->createSession(netConfig);

    float MEMORY = 0.0;
    mnnNet->getSessionInfo(session,MNN::Interpreter::SessionInfoCode::MEMORY, &MEMORY);
    float FLOPS = 0.0;
    mnnNet->getSessionInfo(session,MNN::Interpreter::SessionInfoCode::FLOPS, &FLOPS);
    std::cout << MEMORY << " " << FLOPS << std::endl;

//    MNN::Tensor* input0 = mnnNet->getSessionInput(session,NULL);
    const std::map<std::string, MNN::Tensor*> inputs = mnnNet->getSessionInputAll(session);
//    for(auto &k : inputs){
//        std::cout << k.first << std::endl;
//        for(auto &s : k.second->shape()){
//            std::cout << s << std::endl;
//        }
//    }

    cv::Mat image = cv::imread(image_file);
    float w_scale = (float)image.cols / (float)416;
    float h_scale = (float)image.rows / (float)416;
    cv::Mat image_resize;
    cv::resize(image, image_resize, cv::Size(416,416));
    image_resize.convertTo(image_resize, CV_32F, 1.0/255.0);
    cv::Mat image_channels[3];
    cv::split(image_resize, image_channels);


    auto nchw_tensor = new MNN::Tensor(inputs.at("images"), MNN::Tensor::CAFFE);
    auto nchw_data   = nchw_tensor->host<float>();
    for (int j = 0; j < 3; j++) {
        memcpy(nchw_data + 416*416 * j, image_channels[j].data,416*416 * sizeof(float));
    }

    inputs.at("images")->copyFromHostTensor(nchw_tensor);

    mnnNet->runSession(session);

    const std::map<std::string, MNN::Tensor*> outputs = mnnNet->getSessionOutputAll(session);
    auto output_tensor = outputs.at("output");
    auto output = output_tensor->host<float>();
//    std::cout << output[0] << std::endl;
//    for(auto &k : outputs){
//        std::cout << k.first << std::endl;
//        for(auto &s : k.second->shape()){
//            std::cout << s << std::endl;
//        }
//    }
    std::vector<BoxInfo> boxes;
    std::vector<YoloXAnchor> anchors;
    std::vector<int> strides = {8, 16, 32};
    generate_anchors(416, 416, strides, anchors);
    int num_classes = 80;
    float score_threshold = 0.2;
    for (unsigned int i = 0; i < 3549; ++i){
        const float *offset_obj_cls_ptr = output + (i * (num_classes + 5)); // row ptr
        float obj_conf = offset_obj_cls_ptr[4];
        if (obj_conf < score_threshold) continue; // filter first.

        float cls_conf = offset_obj_cls_ptr[5];
        unsigned int label = 0;
        for (unsigned int j = 0; j < num_classes; ++j){
            float tmp_conf = offset_obj_cls_ptr[j + 5];
            if (tmp_conf > cls_conf){
                cls_conf = tmp_conf;
                label = j;
            }
        } // argmax

        float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
        if (conf < score_threshold) continue; // filter

        const int grid0 = anchors.at(i).at(0);
        const int grid1 = anchors.at(i).at(1);
        const int stride = anchors.at(i).at(2);

        float dx = offset_obj_cls_ptr[0];
        float dy = offset_obj_cls_ptr[1];
        float dw = offset_obj_cls_ptr[2];
        float dh = offset_obj_cls_ptr[3];

        float cx = (dx + (float) grid0) * (float) stride;
        float cy = (dy + (float) grid1) * (float) stride;
        float w = std::exp(dw) * (float) stride;
        float h = std::exp(dh) * (float) stride;
        float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
        float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
        float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
        float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

        types::Boxf box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) img_width - 1.f);
        box.y2 = std::min(y2, (float) img_height - 1.f);
        box.score = conf;
        box.label = label;
        box.label_text = class_names[label];
        box.flag = true;
        bbox_collection.push_back(box);

        count += 1; // limit boxes for nms.
        if (count > max_nms)
            break;
    }

    std::vector<BoxInfo> boxes;
    for(int i = 0; i < 3549; i++){

        if(*(output+i*85+4) > 0.2){
            int cur_label = 0;
            float score = *(output+i*85+4+1);
            for (int label = 0; label < 80; label++)
            {
                //LOGD("decode_infer label %d",label);
                //LOGD("decode_infer score %f",scores[label]);
                if (*(output+i*85+5+label) > score)
                {
                    score = *(output+i*85+5+label);
                    cur_label = label;
                }
            }

            float x = *(output+i*85+0)* 416.0f * w_scale;
//            float x = *(output+i*85+0)*  w_scale;
            float y = *(output+i*85+1)* 416.0f * h_scale;
//            float y = *(output+i*85+1)* h_scale;
            float w = *(output+i*85+2)* 416.0f * w_scale;
//            float w = *(output+i*85+2)*  w_scale;
            float h = *(output+i*85+3)* 416.0f * h_scale;
//            float h = *(output+i*85+3)*  h_scale;

            boxes.push_back(BoxInfo{
                    (float)std::max(0.0, x-w/2.0),
                    (float)std::max(0.0, y-h/2.0),
                    (float)std::min((float)image.cols, (float)(x+w/2.0)),
                    (float)std::min((float)image.rows, (float)(y+h/2.0)),
                    *(output+i*85+4),
                    cur_label
            });
//            std::cout << " x1: " << (float)std::max(0.0, x-w/2.0) <<
//            " y1: " << (float)std::max(0.0, y-h/2.0) <<
//            " x2: " << (float)std::min(320.0, x+w/2.0) <<
//            " y2: " << (float)std::min(320.0, y+h/2.0) <<
//            " socre: " << *(output+i*85+4) <<
//            " label: " << cur_label << std::endl;
        }
    }

    nms(boxes, 0.6);
    for(auto &box: boxes){
        std::cout << " x1: " << box.x1 <<
                  " y1: " << box.y1 <<
                  " x2: " << box.x2 <<
                  " y2: " << box.y2 <<
                  " socre: " << box.score <<
                  " label: " << box.label << std::endl;
    }
    draw_coco_bboxes(image, boxes);
    cv::waitKey(0);


    return 0;
}
