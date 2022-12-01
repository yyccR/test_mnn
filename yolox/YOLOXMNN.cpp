//
// Created by cv on 2021/10/6.
//

#include "YOLOXMNN.h"
#include <vector>

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


YOLOXMNN::YOLOXMNN()
{

}
YOLOXMNN::~YOLOXMNN()
{

}
void  YOLOXMNN::GenGridBox(const int netWidth,
                            const int netHeight)
{
    for (int i = 0; i < 3; i++) {
        int gridRow = int((float )netHeight / mStrides[i]);
        int gridCol = int((float )netWidth / mStrides[i]);
        for(int row = 0; row < gridRow; row++)
        {
            for(int col = 0; col < gridCol; col++)
            {
                GridInfo gridInfo;
                gridInfo.gridX = (float)col;
                gridInfo.gridY = (float)row;
                gridInfo.stride = mStrides[i];
                mGridInfos.push_back(gridInfo);
            }
        }
    }
}
cv::Mat YOLOXMNN::PreprocImage(const cv::Mat& inputImage,
                               const int netWidth,
                               const int netHeight,
                               float& fRatio)
{
//    int width = inputImage.cols, height = inputImage.rows;
//    cv::Mat imageOut(netHeight, netWidth, CV_8UC3);
//    if(width == netWidth && height == netHeight)
//    {
//        inputImage.copyTo(imageOut);
//        return inputImage;
//    }
//    memset(imageOut.data, 114, netWidth * netHeight * 3);
//    fRatio = std::min((float)netWidth /(float)width, (float)netHeight / (float)height);
//    int newWidth = (int)(fRatio * (float )width), newHeight = (int)(fRatio * (float )height);
    cv::Mat rzImage;
//    cv::resize(inputImage, rzImage, cv::Size(newWidth, newHeight));
    cv::resize(inputImage, rzImage, cv::Size(netWidth, netHeight));
//    cv::Mat rectImage = imageOut(cv::Rect(0, 0, newWidth, newHeight));
//    rzImage.copyTo(rectImage);
//    return imageOut;
    return rzImage;

}
void YOLOXMNN::NMS(std::vector<DetBoxes>& detBoxes,
                   std::vector<int>& picked)
{
    std::sort(detBoxes.begin(), detBoxes.end(),
              [](const DetBoxes& a, const DetBoxes& b)
              {
                  return a.scoreObj > b.scoreObj;
              });
    picked.clear();
    const int n = (int)detBoxes.size();
    for (int i = 0; i < n; i++) {
        const DetBoxes &a = detBoxes[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++) {
            const DetBoxes &b = detBoxes[picked[j]];
            // intersection over union
            float  x0 = std::max(a.x,  b.x);
            float  y0 = std::max(a.y, b.y);
            float  x1 = std::min(a.x + a.w, b.x + b.w);
            float  y1 = std::min(a.y + a.h, b.y + b.h);
            float inter_area = std::max(0.0f, (x1 - x0)) * std::max(0.0f , (y1 - y0));
            float union_area = a.area + b.area - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > mNMSThre)
                keep = 0;
        }
        if (keep)
            picked.push_back(i);
    }

}
bool  YOLOXMNN::LoadWeight(const char* weightFile)
{
    mNet.reset(MNN::Interpreter::createFromFile(weightFile));
    if(mNet == nullptr)
    {
        return false;
    }
    mSession = mNet->createSession(mConfig);
    // input tensor config
    mInputTensor = mNet->getSessionInput(mSession, NULL);
    std::vector<int> inputShape = mInputTensor->shape();
    mNetChannel = inputShape[1];
    mNetHeight = inputShape[2];
    mNetWidth = inputShape[3];
    MNN_PRINT("input: w:%d , h:%d, c: %d\n", mNetWidth, mNetHeight, mNetChannel);
    this->GenGridBox(mNetWidth, mNetHeight);
    MNN_PRINT("GRID SIZE: %d \n", (int)mGridInfos.size());

    // image config
    mImageConfig.filterType = MNN::CV::BILINEAR;
    mImageConfig.sourceFormat = MNN::CV::BGR;
    mImageConfig.destFormat = MNN::CV::BGR;

    MNN::CV::Matrix trans;
    trans.setScale(1.0f, 1.0f);
    mPretreat.reset(MNN::CV::ImageProcess::create(mImageConfig));
    mPretreat->setMatrix(trans);

    return true;
}
bool YOLOXMNN::Inference(const cv::Mat& inputImage, std::vector<DetBoxes>& detBoxes, std::vector<BoxInfo>& boxes)
{
    if(!mSession)
    {
        return false;
    }
    float ratio = 0;
    cv::Mat netImage = this->PreprocImage(inputImage, mNetWidth, mNetHeight, ratio);
//    mPretreat->convert((uint8_t*)netImage.data, netImage.cols, netImage.rows, 0, mInputTensor);
    netImage.convertTo(netImage, CV_32F, 1.0);
    cv::Mat image_channels[3];
    cv::split(netImage, image_channels);
    auto nchw_tensor = new MNN::Tensor(mInputTensor, MNN::Tensor::CAFFE);
    auto nchw_data   = nchw_tensor->host<float>();
    for (int j = 0; j < 3; j++) {
        memcpy(nchw_data + 416*416 * j, image_channels[j].data,416*416 * sizeof(float));
    }
    mInputTensor->copyFromHostTensor(nchw_tensor);

    mNet->runSession(mSession);
    MNN::Tensor* outputTensor = mNet->getSessionOutput(mSession, NULL);
    this->Postprocess(outputTensor, ratio, detBoxes, boxes);
    return true;
}

void YOLOXMNN::Postprocess(const MNN::Tensor* outTensor,
                           const float  ratio,
                            std::vector<DetBoxes>& outBoxes, std::vector<BoxInfo>& boxes)
{
    outBoxes.clear();
    int outHW = 0, outChannel = 0;
    std::vector<int> outShape = outTensor->shape();
    outHW = outShape[1];
    outChannel = outShape[2];
    MNN_PRINT("output: wh: %d, c: %d \n", outHW, outChannel);
    MNN::Tensor outTensorHost(outTensor, outTensor->getDimensionType());
    outTensor->copyToHostTensor(&outTensorHost);
//    float* outData = outTensorHost.host<float>();
//    MNN_PRINT("outData: index:0 , value: %.2f \n", outData[0]);
    float* output = outTensorHost.host<float>();

    std::vector<DetBoxes> detBoxes;

    int num_classes = 80;
    float score_threshold = 0.3;
    std::vector<YoloXAnchor> anchors;
    std::vector<int> strides = {8, 16, 32};
    generate_anchors(416, 416, strides, anchors);
    for (unsigned int i = 0; i < 3549; ++i){
        DetBoxes  detBox;
        const float *offset_obj_cls_ptr = output + (i * (num_classes + 5)); // row ptr
        float obj_conf = *(offset_obj_cls_ptr+4);
        if (obj_conf < score_threshold) continue; // filter first.

        float cls_conf = *(offset_obj_cls_ptr+5);
        int label = 0;
        for (int j = 0; j < num_classes; ++j){
//            float tmp_conf = offset_obj_cls_ptr[j + 5];
            float tmp_conf = *(offset_obj_cls_ptr+5+j);
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

//        float dx = offset_obj_cls_ptr[0];
        float dx = *(offset_obj_cls_ptr+0);
//        float dy = offset_obj_cls_ptr[1];
        float dy = *(offset_obj_cls_ptr+1);
//        float dw = offset_obj_cls_ptr[2];
        float dw = *(offset_obj_cls_ptr+2);
//        float dh = offset_obj_cls_ptr[3];
        float dh = *(offset_obj_cls_ptr+3);

        float cx = (dx + (float) grid0) * (float) stride;
        float cy = (dy + (float) grid1) * (float) stride;
        float w = std::exp(dw) * (float) stride;
        float h = std::exp(dh) * (float) stride;
        float x1 = (cx - w / 2.f);
        float y1 = (cy - h / 2.f);
        float x2 = (cx + w / 2.f);
        float y2 = (cy + h / 2.f);

        BoxInfo box;
        box.x1 = std::max(0.f, x1);
        box.y1 = std::max(0.f, y1);
        box.x2 = std::min(x2, (float) 416 - 1.f);
        box.y2 = std::min(y2, (float) 416 - 1.f);
        box.score = conf;
        box.label = label;
        boxes.push_back(box);


        detBox.x = box.x1;
        detBox.y = box.y1;
        detBox.w = w;
        detBox.h = h;
        detBox.iouScore = obj_conf;
        detBox.score = cls_conf;
        detBox.clsIndex = box.label;
        detBox.area = w*h;
        detBox.scoreObj = detBox.score * detBox.iouScore;
        detBoxes.push_back(detBox);

    }

//    std::vector<DetBoxes> detBoxes;
//    for (int i = 0; i < outHW; ++i, outData+=outChannel) {
//        DetBoxes  detBox;
//        // decoder
//        float  centerX = (mGridInfos[i].gridX + outData[0]) * mGridInfos[i].stride;
//        float  centerY = (mGridInfos[i].gridY + outData[1]) * mGridInfos[i].stride;
//        detBox.w = std::exp(outData[2]) * mGridInfos[i].stride;
//        detBox.h = std::exp(outData[3]) * mGridInfos[i].stride;
//        detBox.x = centerX - detBox.w * 0.5f;
//        detBox.y = centerY - detBox.h * 0.5f;
//        detBox.iouScore = outData[4];
//        float score = 0.0f;
//        int clsIndex = 0;
//        float* clsScoreData = outData + 5;
//        for (int j = 0; j < (outChannel - 5); ++j) {
//            if(score < clsScoreData[j])
//            {
//                score = clsScoreData[j];
//                clsIndex = j;
//            }
//        }
//        detBox.score = score;
//        detBox.clsIndex = clsIndex;
//        detBox.area = detBox.w * detBox.h;
//        detBox.scoreObj = detBox.score * detBox.iouScore;
//        if(detBox.scoreObj >= mClsThre) {
//            detBoxes.push_back(detBox);
//        }
//    }

    nms(boxes, 0.3);
//    for(auto &box: boxes){
//        std::cout << " x1: " << box.x1 <<
//                  " y1: " << box.y1 <<
//                  " x2: " << box.x2 <<
//                  " y2: " << box.y2 <<
//                  " socre: " << box.score <<
//                  " label: " << box.label << std::endl;
//    }
//    draw_coco_bboxes(image, boxes);
//    cv::waitKey(0);

    std::vector<int> picked;
    this->NMS(detBoxes, picked);
    for (int i = 0; i < (int) picked.size() ; ++i) {
        DetBoxes&  detPickedBox = detBoxes[picked[i]];
        detPickedBox.x /= ratio;
        detPickedBox.y /= ratio;
        detPickedBox.w /= ratio;
        detPickedBox.h /= ratio;
        outBoxes.push_back(detPickedBox);
    }
    printf("det num: %d \n",(int) picked.size());

}
