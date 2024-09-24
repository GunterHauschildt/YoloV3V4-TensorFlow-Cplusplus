#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/chrono.hpp>

#include <unordered_map>

#include "..\Include\Yolo_v3v4.h"
#include "..\Include\json.h"

#ifdef WINDOWS_BUILD
#include "..\Include\Log.h"
#include "..\Include\ImageProcess.h"
#endif

using namespace std;
using namespace cv;
using namespace dnn;
using namespace xt;


YoloV3V4_Inference::YoloV3V4_Inference (
	const std::vector<std::string>& channelNames_,
	const std::vector<int> imageIndexs_, 
	const cv::Vec3i& imageSize_,
	const bool allowResize_,
    int numEngineClasses,
	std::unordered_map <std::string, ObjectDetectionObjectInfo>& objectInfos_,	
    const std::string& root,
	const std::string& enginePath_, 
	const std::string& engineFileName_,
	const std::string& anchorsFileName_,
	const DNNTargets::CPU_OPENCL_CUDA target_,
	const float nmsThreshold_
	)
{
	this->channelNames = vector<string>(channelNames_);
	this->imageIndexs = vector<int>(imageIndexs_);
	this->imageSizeAndDims = Vec3i(imageSize_);
	this->imageSize = Size(imageSize_[0], imageSize_[1]);
	this->objectInfos = std::unordered_map <std::string, ObjectDetectionObjectInfo>(objectInfos_);
	this->nmsThreshold = nmsThreshold_;
	this->allowResize = allowResize_;

	for (auto& objectInfo : objectInfos)
	{
		if (this->objectNameLUT.count(objectInfo.second.ID))
			Log(Log::FatalError) << "Found two objects with the same ID. In: " << __FUNCTION__ << Log::Submit;
		this->objectNameLUT[objectInfo.second.ID] = objectInfo.second.name;
	}

	// string root = MasterConfig::getCustomerDirectory();
	string engineFilePath  = root + "\\" + enginePath_ + "\\" + engineFileName_;
	string anchorsFilePath = root + "\\" + enginePath_ + "\\" + anchorsFileName_;


	if (!boost::filesystem::is_regular_file(engineFilePath))
		Log(Log::FatalError) 
			<< "Couldn't open: " << engineFilePath << "In: " <<  __FUNCTION__ << " " << __LINE__ << Log::Submit;

	if (!boost::filesystem::is_regular_file(anchorsFilePath))
		Log(Log::FatalError) 
			<< "Couldn't open: " << anchorsFilePath << "In: " <<  __FUNCTION__ << " " << __LINE__ << Log::Submit;

	//////////

	float nmsConfidence = std::numeric_limits<float>::max();
	for (auto& objectInfo : objectInfos)
	{
		if (objectInfo.second.confidence < nmsConfidence)
			nmsConfidence = objectInfo.second.confidence;
	}

	spYolo = shared_ptr<Yolo_v3v4>(new Yolo_v3v4(
		numEngineClasses,
		1,
		imageSizeAndDims[1], imageSizeAndDims[0], imageSizeAndDims[2],
		engineFilePath,
		anchorsFilePath,
		target_,
		nmsConfidence, 
		nmsThreshold
	));
}

xt::xarray<float> xt_sigmoid(const xt::xarray<float>& in)
{
    return xt::xarray<float>(1.0f / (1.0f + xt::exp(-in)));
}


static void splitAndCorrectConfidencesAndClassProbs(
    const xt::xarray <float>& decodedObjectnesses, 
    const xt::xarray <float>& decodedClassProbs, 
    int numClasses, 
    std::vector <std::vector<float>>& confidences,
    std::vector <std::vector<std::vector<float>>>& classProbs
    )
{
    int batchSize = (int) decodedObjectnesses.shape()[0];
    int numBoxes = (int) decodedClassProbs.shape()[1];

    xarray <float> xConfidences;

    xConfidences = decodedObjectnesses;

    confidences.resize(batchSize); 
    classProbs.resize(batchSize);
    for (int image=0, cv=0; image<batchSize; image++)
    {
        confidences.at(image).resize(numBoxes);
        classProbs.at(image).resize(numBoxes);
        for (int b=0; b<numBoxes; b++)
        {
            classProbs.at(image).at(b).resize(numClasses);
            confidences.at(image).at(b) = xConfidences.at(image, b, 0);
            for (int c=0; c<numClasses; c++)
                classProbs.at(image).at(b).at(c) = decodedClassProbs.at(image, b, c);
        }
    }
}

static void splitAndCorrectBoxes(const xt::xarray<float>& boxesRaw, const cv::Size& imageSize, std::vector <std::vector <cv::Rect>>& boxes)
{
    int batchSize = (int) boxesRaw.shape()[0];
    int numBoxes = (int) boxesRaw.shape()[1];

    boxes.resize(batchSize);
    for (int image=0; image<batchSize; image++)
    {
        boxes.at(image).resize(numBoxes);
        for (int box=0; box<boxesRaw.shape()[1]; box++)
        {
                boxes.at(image).at(box) = Rect(
                Point(
                    cvRound(boxesRaw.at(image, box, 0) * imageSize.width),
                    cvRound(boxesRaw.at(image, box, 1) * imageSize.height)),
                Point(
                    cvRound(boxesRaw.at(image, box, 2) * imageSize.width),
                    cvRound(boxesRaw.at(image, box, 3) * imageSize.height))
            );
        }
    }
}

static void decodeYoloPredictions(
    const xt::xarray<float>& pred, 
    const xt::xarray<float>& anchors, 
    const int numClasses,
    xt::xarray<float>& boxes_x0y0x1y1,
    xt::xarray<float>& objectnesses, 
    xt::xarray<float>& class_probs
)
{
    // doesn't return pred_box like python (only needed for training)
    // to do: why did tf.meshgrid work?? (in python yolo_boxes) (maybe it didn't !!)
    // to do: why does tf.meshgrid work?? (in python yolo_loss)

    try {
        // to do: don't know about the need for all these xarray copy constructors from xview !!

        auto grid_size_xy = xarray<int>{ (int) pred.shape()[2], (int)  pred.shape()[1] };
        auto boxes_xy = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(0, 2)));
        auto boxes_wh = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(2, 4)));

        objectnesses = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(4, 5)));
        class_probs = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::all(), xt::all(), xt::range(5, 5+numClasses)));

        boxes_xy = xt_sigmoid(boxes_xy);
        objectnesses = xt_sigmoid(objectnesses);
        class_probs = xt_sigmoid(class_probs);

        auto grid_x = xt::arange<int>(grid_size_xy[0]);
        auto grid_xy_x = xt::reshape_view(
            xt::tile(grid_x, grid_size_xy[1]),
            { grid_size_xy[1], grid_size_xy[0] }
        );

        auto grid_y = xarray<int>(xt::arange<int>(grid_size_xy[1]));
        auto grid_xy_y = xt::reshape_view(
            xt::repeat(grid_y, grid_size_xy[0], 0),
            { grid_size_xy[1], grid_size_xy[0] }
        );

        auto grid_xy = xarray<int>(xt::stack(xt::xtuple(grid_xy_x, grid_xy_y), 2));
        grid_xy = xt::expand_dims(grid_xy, 2);

        auto grid_xy_f = xt::cast<float>(grid_xy);
        auto grid_size_xy_f = xt::cast<float>(grid_size_xy);

        boxes_xy = (boxes_xy + grid_xy_f) / grid_size_xy_f;

        boxes_wh = xt::exp(boxes_wh) * anchors;

        auto boxes_x0y0 = boxes_xy - (boxes_wh / 2.0f);
        auto boxes_x1y1 = boxes_xy + (boxes_wh / 2.0f);
        boxes_x0y0x1y1 = xt::concatenate(xt::xtuple(boxes_x0y0, boxes_x1y1), 4);

    }
    catch (std::exception& e) {
        Log(Log::Warning) << "S/W error: " << e.what() << ". In: " << __FUNCTION__ << Log::Submit;
    }
    catch (...) {
        Log(Log::Warning) << "Unknown error. In: " << __FUNCTION__ << Log::Submit;
    }
}

std::unordered_map <std::string, xt::xarray<float>> readYoloAnchors(const std::string& fileName)
{
    auto anchors = unordered_map <string, xt::xarray<float>>();
        
	try {

		ifstream jsonFile(fileName);
		
		if (jsonFile.good() == false)
		{
			jsonFile.close();
			return std::unordered_map <std::string, xt::xarray<float>>();
    	}
		else
		{
			stringstream jsonStream;
			jsonStream << jsonFile.rdbuf();
			jsonFile.close();

			json::Value infoDeSered;
			try {
			    infoDeSered = json::Deserialize(jsonStream.str());
			} catch(...) {
				cout << "Can't parse: "<< fileName  << endl;
				return std::unordered_map <std::string, xt::xarray<float>>();
			}
			
			json::Object json;
			try {
				json = json::Object(infoDeSered);
			} catch(...) {
				cout << "Can't parse: "<< fileName  << endl;
				return std::unordered_map <std::string, xt::xarray<float>>();
			}

            vector <string> yoloOutputs = {
                "yolo_output_0", "yolo_output_1", "yolo_output_2"
            };

            for (auto& yoloOutput : yoloOutputs)
            {
			    try {
				    int numAnchors = (int) json["anchors"][yoloOutput].ToArray().size();
                    anchors[yoloOutput] = xt::empty<float>({ numAnchors, 2});
                    
                    for (int a=0; a<numAnchors; a++)
                    {
                        anchors[yoloOutput].at(a, 0) = json["anchors"][yoloOutput].ToArray()[a].ToArray()[0].ToFloat();
                        anchors[yoloOutput].at(a, 1) = json["anchors"][yoloOutput].ToArray()[a].ToArray()[1].ToFloat();
                    }
 
			    } catch(...) {
		    	}
            }
		}
    }
    catch(...) {
        // error handling TBD
    }
    return anchors;
}

std::shared_ptr <std::mutex> Yolo_v3v4::m_Mutex = shared_ptr<mutex>(new mutex);

Yolo_v3v4::Yolo_v3v4(
    int numClasses, 
    int batchSize, 
    int imageHeight, int imageWidth, int imageChannels,
    const std::string& onnxPath,
    const std::string& anchorsPath,
    DNNTargets::CPU_OPENCL_CUDA target,
    float nmsConfidence,
    float nmsThreshold
)
{
    m_BatchSize = batchSize;
    m_NumClasses = numClasses;
    m_nmsConfidence = nmsConfidence;
    m_nmsThreshold = nmsThreshold;

    // read net

    try {

        m_Net = readNetFromONNX(onnxPath);

    } catch(cv::Exception& e) {
		Log(Log::FatalError) 
            << "Opencv Error calling readNetFromONNX(): " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::FatalError) 
            << "S/W Error calling readNetFromONNX(): " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::FatalError) 
            << "Unknown S/W Error calling readNetFromONNX() in: " <<  __FUNCTION__ << Log::Submit;
	}

    // set target, backend

#ifdef _DEBUG
    target = CPU;
#endif

    try {

        if (target == OPENCL)
        {
            m_Net.setPreferableTarget(DNN_TARGET_OPENCL);
            m_Net.setPreferableBackend(DNN_BACKEND_OPENCV);
        }
        else if (target == CUDA)
        {
            m_Net.setPreferableTarget(DNN_TARGET_CUDA);
            m_Net.setPreferableBackend(DNN_BACKEND_CUDA);
        }
        else if (target == CPU)
        {
            m_Net.setPreferableTarget(DNN_TARGET_CPU);
            m_Net.setPreferableBackend(DNN_BACKEND_OPENCV);
        }

    } catch(cv::Exception& e) {
		Log(Log::FatalError) 
            << "Opencv Error calling setPreferableTarget() / setPreferableTarget(): " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::FatalError) 
            << "S/W Error calling setPreferableTarget() / setPreferableTarget(): " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::FatalError) 
            << "Unknown S/W Error calling setPreferableTarget() / setPreferableTarget() "
            << ". In " <<  __FUNCTION__ << Log::Submit;
    }

    // set input layers and input size for convience later

    try {

        m_InputLayers.resize(1, "input"); 
        m_InputShapes[m_InputLayers.at(0)] = MatShape({ batchSize, imageHeight, imageWidth, imageChannels });

        m_InputImageSize = Size(m_InputShapes["input"].at(2),  m_InputShapes["input"].at(1));
        m_InputImageChannels = m_InputShapes["input"].at(3);

    } catch(cv::Exception& e) {
		Log(Log::FatalError) 
            << "Opencv Error calling setting input layers: " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::FatalError) 
            << "S/W Error calling setting input layers: " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::FatalError) << "Unknown S/W Error setting input layers in: " <<  __FUNCTION__ << Log::Submit;
	}
    
    // set anchors as per file

    vector <string> anchorsOutputs;
    std::unordered_map <std::string, xt::xarray<float>> anchors;
    try {

        anchors = readYoloAnchors(anchorsPath);
        for (auto& anchor : anchors)
            anchorsOutputs.push_back(anchor.first);

    } catch(cv::Exception& e) {
		Log(Log::FatalError) 
            << "Opencv Error reading anchors: " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::FatalError) 
            << "S/W Error reading anchors: " << e.what() 
            << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::FatalError) << "Unknown S/W Error reading anchors in: " <<  __FUNCTION__ << Log::Submit;
	}

    // read output layers ...

    try {

        m_OutputLayers = m_Net.getUnconnectedOutLayersNames();
    
    } catch(cv::Exception& e) {
		Log(Log::FatalError) 
            << "Opencv Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::FatalError) << "S/W Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::FatalError) << "Unknown S/W Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << "In " <<  __FUNCTION__ << Log::Submit;
	}

    // verify anchors match output layers

    for (auto& anchor: anchorsOutputs)
    {
        if (!inVector(anchor, m_OutputLayers))
            Log(Log::FatalError) 
                << "Anchor: " << anchor << " not a Yolo output. " << anchorsPath
                << Log::Submit;
    }

    // verify output layers match anchors

    for (auto& outputLayer : m_OutputLayers)
    {
        if (!inVector(outputLayer, anchorsOutputs))
            Log(Log::FatalError) 
                << "OutputLayer: " << outputLayer << " not in anchors file: " << anchorsPath
                << ", but found in onnx model. In: " << __FUNCTION__ 
                << Log::Submit;
    }

    // a little confusing but we get the output shapes from the input shape
    // output shapes are the dimensions of the prediction results so we can decode from
    // the 5D prediction array into boxes, confidences, etc

    for (auto& outputLayer : m_OutputLayers)
    {
        vector <dnn::MatShape> inLayerShapes;
        vector <dnn::MatShape> outLayerShapes;
        try {

            int id = m_Net.getLayerId(outputLayer);
            m_Net.getLayerShapes(m_InputShapes["input"], id, inLayerShapes, outLayerShapes);

        } catch(cv::Exception& e) {
		    Log(Log::FatalError) 
                << "Opencv Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	    } catch(std::exception& e) {
		    Log(Log::FatalError) << "S/W Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	    } catch(...) {
		    Log(Log::FatalError) << "Unknown S/W Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << "In " <<  __FUNCTION__ << Log::Submit;
	    }

        // don't think this can ever occur ...

        if (outLayerShapes.size() != 1)
		    Log(Log::FatalError) 
                << "Received an incorrect outLayerShape. Check onnx_model and ensure a valid yolo v3 or v4 model. "
                << "In " <<  __FUNCTION__ << Log::Submit;

        // set the shapes, anchors as per 'YoloBoxes' in the model

        try {
           
            m_InputShapes[outputLayer] = outLayerShapes.at(0);
            m_Anchors[outputLayer] = anchors[outputLayer] / xt::xarray<float>({ (float)imageWidth, (float)imageHeight });

        } catch(cv::Exception& e) {
		    Log(Log::FatalError) 
                << "Opencv Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	    } catch(std::exception& e) {
		    Log(Log::FatalError) 
                << "S/W Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	    } catch(...) {
		    Log(Log::FatalError)
                << "Unknown S/W Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << "In " <<  __FUNCTION__ << Log::Submit;
	    }
    }
};

void Yolo_v3v4::forward(const cv::Mat& batch)
{
    this->forward(vector<Mat>(1, batch));
}


void Yolo_v3v4::forward(const std::vector <cv::Mat>& images)
{
    try {

        m_Predictions_xt.clear();
        m_CorrectedBoxes.clear();
        m_Confidences.clear();
        m_Valid.clear();
        m_ClassProbs.clear();

        if (!m_BatchSize || (images.size() != m_BatchSize))
            return;

        // resizing control handled externally
        for (auto& image : images)
        {
            if ((image.size() != m_InputImageSize) || (image.channels() != m_InputImageChannels))
            {
                stringstream ss1, ss2;
                ss1 << image.size() << ", " << image.channels();
                ss2 << m_InputImageSize << ", " << m_InputImageChannels;

                Log(Log::Warning) 
                    << "Image size (" << ss1.str() << ") does not match network size (" << ss2.str() << ") ." 
                    << "In: " << __FUNCTION__ << Log::Submit;
                return;
            }
        }

        // copy to batch by changing color (BGR2RGB for TF network) into it's ROI
        // then convert the whole batch to float
 
        Mat batch = Mat(m_InputImageSize.height * m_BatchSize, m_InputImageSize.width, images.at(0).type());
        for (auto& image : images)
        {
            int i = (int) (&image - &images[0]);
            Rect ROI = Rect(Point(0, (i+0)*image.rows), Point(image.cols, (i+1)*image.rows));
            cvtColor(image, batch(ROI), cv::COLOR_BGR2RGB);
        }
        batch.convertTo(batch, CV_32F, 1.0f/255.0f);

        int inputShape[4] = { 
            m_InputShapes["input"].at(0), 
            m_InputShapes["input"].at(1), 
            m_InputShapes["input"].at(2), 
            m_InputShapes["input"].at(3) 
        };

        Mat blob = Mat(4, inputShape, CV_32F, batch.data);

        { std::lock_guard <std::mutex> guard(*m_Mutex);
          m_Net.setInput(blob);
          m_Net.forward(m_Predictions_cv, m_OutputLayers);
        }

    } catch(cv::Exception& e) {
		Log(Log::Warning) << "Opencv Error: " << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::Warning) << "S/W Error: " << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::Warning) << "Unknown S/W Error in " <<  __FUNCTION__ << Log::Submit;
	}
}

void Yolo_v3v4::decode()
{
    try {

        m_Predictions_xt.clear();
        m_CorrectedBoxes.clear();
        m_Confidences.clear();
        m_Valid.clear();
        m_ClassProbs.clear();

        m_Predictions_xt = vector<xarray<float>>(m_OutputLayers.size());

        vector <xarray<float>> boxes_x0y0x1y1(m_OutputLayers.size());
        vector <xarray<float>> objectnesses(m_OutputLayers.size()); 
        vector <xarray<float>> class_probs(m_OutputLayers.size());
 
        for (int o=0; o<m_OutputLayers.size(); o++)
        {
            auto& output = m_OutputLayers.at(o);

            m_Predictions_xt.at(o) = xarray<float>::from_shape({ m_Predictions_cv.at(o).total() });

            // to do: remove a couple of steps in particular the copy
            // (but based on profile the copy won't make a sig difference)

            vector <float> probs_v;
            probs_v.assign((float*)m_Predictions_cv.at(o).datastart, (float*)m_Predictions_cv.at(o).dataend);

            for (int i = 0; i < m_Predictions_cv.at(o).total(); i++)
                m_Predictions_xt.at(o).at(i) = probs_v.at(i);

            //int numPredictClasses = m_InputShapes[output].at(4) - 5;
            //if (!(m_NumClasses == numPredictClasses))
            //{
            //    Log(Log::Warning) 
            //        << __FUNCTION__ << " num classes (" << m_NumClasses << ")  not equal to prediction size (" 
            //        << numPredictClasses << "). " << Log::Submit;
            //    continue;
            //}

            m_Predictions_xt.at(o).resize({
                (unsigned long long) m_InputShapes[output].at(0),
                (unsigned long long) m_InputShapes[output].at(1),
                (unsigned long long) m_InputShapes[output].at(2),
                (unsigned long long) m_InputShapes[output].at(3),
                (unsigned long long) m_InputShapes[output].at(4)
            });

            decodeYoloPredictions(
                m_Predictions_xt.at(o), 
                m_Anchors[output], 
                m_NumClasses,
                boxes_x0y0x1y1.at(o),
                objectnesses.at(o),
                class_probs.at(o)
            );
        }    
 
        // reshape so we can concatenate
        // to do: why can't i use -1 syntax ala numpy

        for (int o=0; o< boxes_x0y0x1y1.size(); o++)
        {
            xarray <float>& decodedBoxes = boxes_x0y0x1y1.at(o);
            xarray <float>& decodedObjectnesses = objectnesses.at(o);
            xarray <float>& decodedClassProbs = class_probs.at(o);
                    
            decodedBoxes =  decodedBoxes.reshape(
                { decodedBoxes.shape()[0], 
                    decodedBoxes.shape()[1] * decodedBoxes.shape()[2] * decodedBoxes.shape()[3],
                    decodedBoxes.shape()[4]
                } 
            );

            decodedObjectnesses = decodedObjectnesses.reshape(
                { decodedObjectnesses.shape()[0], 
                    decodedObjectnesses.shape()[1] * decodedObjectnesses.shape()[2] * decodedObjectnesses.shape()[3],
                    decodedObjectnesses.shape()[4]
                }
            );

            decodedClassProbs =  decodedClassProbs.reshape(
                { decodedClassProbs.shape()[0], 
                    decodedClassProbs.shape()[1] * decodedClassProbs.shape()[2] * decodedClassProbs.shape()[3],
                    decodedClassProbs.shape()[4] }
            );
        }

        // concatenate as required

        xarray <float>& decodedBoxes = boxes_x0y0x1y1.at(0);
        xarray <float>& decodedObjectnesses = objectnesses.at(0);
        xarray <float>& decodedClassProbs = class_probs.at(0);

        for (int o=1; o< boxes_x0y0x1y1.size(); o++)
        {
            decodedBoxes = xt::concatenate(xt::xtuple(decodedBoxes, boxes_x0y0x1y1.at(o)), 1);
            decodedObjectnesses = xt::concatenate(xt::xtuple(decodedObjectnesses, objectnesses.at(o)), 1);
            decodedClassProbs = xt::concatenate(xt::xtuple(decodedClassProbs, class_probs.at(o)), 1);
        }

        splitAndCorrectBoxes(decodedBoxes, 
            Size(m_InputShapes[m_InputLayers.at(0)][2],  m_InputShapes[m_InputLayers.at(0)][1]), 
            m_CorrectedBoxes);
        splitAndCorrectConfidencesAndClassProbs(decodedObjectnesses, decodedClassProbs, m_NumClasses, m_Confidences, m_ClassProbs);

        for (int i=0; i<m_BatchSize; i++)
            dnn::NMSBoxes(m_CorrectedBoxes.at(i), m_Confidences.at(i), m_nmsConfidence, m_nmsThreshold, m_Valid);
 
    } catch(cv::Exception& e) {
		Log(Log::Warning) << "Opencv Error: " << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(std::exception& e) {
		Log(Log::Warning) << "S/W Error: " << e.what() << ". In " <<  __FUNCTION__ << Log::Submit;
	} catch(...) {
		Log(Log::Warning) << "Unknown S/W Error in " <<  __FUNCTION__ << Log::Submit;
	}
}


std::vector <std::vector <cv::Rect>>& Yolo_v3v4::boxes()
{
    return m_CorrectedBoxes;
}

std::vector <std::vector <float>>& Yolo_v3v4::confidences()
{
    return m_Confidences;
}

std::vector <std::vector <std::vector<float>>>& Yolo_v3v4::classProbs()
{
    return m_ClassProbs;
}

std::vector <int>& Yolo_v3v4::valid()
{
    return m_Valid;
}
