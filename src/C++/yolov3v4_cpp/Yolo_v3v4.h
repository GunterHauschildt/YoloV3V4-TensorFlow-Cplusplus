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

#include "..\include\DNNTargets.h"

class Yolo_v3v4 : public DNNTargets {
public:
    Yolo_v3v4(
        int numClasses, int batchSize, int imageHeight, int imageWidth, int imageChannels,
        const std::string& onnxPath,
        const std::string& anchorsPath,
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

class ObjectDetectionObjectInfo {
	public:
		ObjectDetectionObjectInfo()
		{
			name = "";
			ID = 0;
			minSize = cv::Size();
			maxSize = cv::Size();
			confidence = 0.50;
			setExceptionFlag = "";
		}
		ObjectDetectionObjectInfo(std::string& name_, int ID_, cv::Size& minSize_, cv::Size& maxSize_, float confidence_, std::string& setExceptionFlag_)
		{
			name = std::string(name_);
			ID = ID_;
			minSize = cv::Size(minSize_);
			maxSize = cv::Size(maxSize_);
			confidence = confidence_;
			setExceptionFlag = std::string(setExceptionFlag_);
		}
		std::string name;
		int ID;
		cv::Size minSize;
		cv::Size maxSize;
		float confidence;
		std::string setExceptionFlag;
	};


class YoloV3V4_Inference {
	public:
		YoloV3V4_Inference() {};
		YoloV3V4_Inference(
			const std::vector <std::string>& channelNames_,
			const std::vector <int> imageIndexs_, 
			const cv::Vec3i& imageSize_,
			bool allowResize_,
			int numEngineClasses,
			std::unordered_map <std::string, ObjectDetectionObjectInfo>& objectInfos_,
			const std::string& root,
			const std::string& yoloPath_, 
			const std::string& yoloFileName_,
			const std::string& anchorsFileName_,
			const DNNTargets::CPU_OPENCL_CUDA target_,
			const float nmsThreshold_
		);
		std::vector <std::string> channelNames;
		std::vector <int> imageIndexs;
		cv::Vec3i imageSizeAndDims;
		cv::Size imageSize;
		int camerasImageNum;
		float nmsThreshold;
		std::shared_ptr <Yolo_v3v4> spYolo;
		std::unordered_map <std::string, ObjectDetectionObjectInfo> objectInfos;
		std::unordered_map <int, std::string> objectNameLUT;
		bool allowResize;
	};

