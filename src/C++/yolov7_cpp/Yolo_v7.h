#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <unordered_map>

#pragma warning(push)
#pragma warning(disable: 4267)
#include "xtensor\xarray.hpp"
#include "xtensor\xpad.hpp"
#include "xtensor\xio.hpp"
#include "xtensor\xview.hpp"
#include "xtensor\xslice.hpp"
#pragma warning(pop)

class DNNTargets {
public:
    enum CPU_OPENCL_CUDA { CPU, OPENCL, CUDA };
};

class DNNOrder {
public:
    enum ORDER { CH_W_H, W_H_CH };
};

class Yolo_v7 : public DNNTargets {
public:
    Yolo_v7(
        int numClasses, int batchSize, int imageHeight, int imageWidth, int imageChannels,
        DNNOrder::ORDER dnnOrder,
        bool swapRB,
        const std::string& onnxPath,
        DNNTargets::CPU_OPENCL_CUDA target,
        float nmsConfidence,
        float nmsThreshold        
    );

    void forward(const cv::Mat& batch);
    void forward(const std::vector <cv::Mat>& batchv);
    void decode();
    std::vector <std::vector <cv::Rect>>& boxes();
    std::vector <std::vector <float>>& confidences();
    std::vector <std::vector <std::vector<float>>>& classProbs();
    std::vector <int>& valid();
    cv::Size& inputImageSize()
    {
        return(m_InputImageSize);
    }
        
private:
    static std::shared_ptr <std::mutex> m_Mutex;
    int m_BatchSize;
    int m_NumClasses;
    float m_nmsThreshold;
    float m_nmsConfidence;
    cv::Size m_InputImageSize;
    int m_InputImageChannels;
    DNNOrder::ORDER m_InputOrder;
    bool m_SwapRB;

    cv::dnn::Net m_Net;
    std::vector <std::string> m_InputLayers;
    std::vector <std::string> m_OutputLayers;
    std::unordered_map <std::string, cv::dnn::MatShape> m_InputShapes;
    std::unordered_map <std::string, xt::xarray <float>> m_Anchors;
    std::vector <cv::Mat> m_Predictions_cv;
    std::vector <xt::xarray <float>> m_Predictions_xt;

    std::vector <std::vector <cv::Rect>> m_CorrectedBoxes;
    std::vector <std::vector <float>> m_Confidences;
    std::vector <std::vector <std::vector<float>>> m_ClassProbs;
    std::vector <int> m_Valid;
};