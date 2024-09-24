#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/filesystem.hpp>
#include <boost/chrono.hpp>

#include <unordered_map>

#include "Yolo_v7.h"

using namespace std;
using namespace cv;
using namespace dnn;
using namespace xt;

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
                    cvRound(boxesRaw.at(image, box, 0)), // * imageSize.width),
                    cvRound(boxesRaw.at(image, box, 1))), //* imageSize.height)),
                Point(
                    cvRound(boxesRaw.at(image, box, 2)), //* imageSize.width),
                    cvRound(boxesRaw.at(image, box, 3))) //* imageSize.height))
            );
        }
    }
}

static void decodeYoloPredictions(
    const xt::xarray<float>& pred, 
    const int numClasses,
    xt::xarray<float>& boxes_x0y0x1y1,
    xt::xarray<float>& confidences, 
    xt::xarray<float>& class_probs
)
{
    try {

        auto boxes_xy = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::range(0, 2)));
        auto boxes_wh = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::range(2, 4)));

        auto boxes_x0y0 = boxes_xy - (boxes_wh / 2.0f);
        auto boxes_x1y1 = boxes_xy + (boxes_wh / 2.0f);
        boxes_x0y0x1y1 = xt::concatenate(xt::xtuple(boxes_x0y0, boxes_x1y1), 2);

        confidences = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::range(4, 5)));
        class_probs = xarray<float>(xt::view(pred, xt::all(), xt::all(), xt::range(5, 5+numClasses)));
    }
    catch (std::exception& e) {
        cout << "Warning: " << "S/W error: " << e.what() << ". In: " << __FUNCTION__ << endl;
    }
    catch (...) {
        cout << "Warning: " << "Unknown error. In: " << __FUNCTION__ << endl;
    }
}


std::shared_ptr <std::mutex> Yolo_v7::m_Mutex = shared_ptr<mutex>(new mutex);

Yolo_v7::Yolo_v7(
    int numClasses, int batchSize, int imageHeight, int imageWidth, int imageChannels,
    DNNOrder::ORDER dnnOrder,
    bool swapRB,
    const std::string& onnxPath,
    DNNTargets::CPU_OPENCL_CUDA target,
    float nmsConfidence,
    float nmsThreshold        
)
{
    m_BatchSize = batchSize;
    m_NumClasses = numClasses;
    m_nmsConfidence = nmsConfidence;
    m_nmsThreshold = nmsThreshold;

    try {

        m_Net = readNetFromONNX(onnxPath);

    } catch(cv::Exception& e) {
		cout << "FATAL ERROR: " 
            << "Opencv Error calling readNetFromONNX(): " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(std::exception& e) {
		cout << "FATAL ERROR: " 
            << "S/W Error calling readNetFromONNX(): " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(...) {
		cout << "FATAL ERROR: " 
            << "Unknown S/W Error calling readNetFromONNX() in: " <<  __FUNCTION__ << endl;
            exit(1);
	}

    m_InputOrder = dnnOrder;
    m_SwapRB = swapRB;


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
		cout << "FATAL ERROR: " 
            << "Opencv Error calling setPreferableTarget() / setPreferableTarget(): " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(std::exception& e) {
		cout << "FATAL ERROR: " 
            << "S/W Error calling setPreferableTarget() / setPreferableTarget(): " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(...) {
		cout << "FATAL ERROR: " 
            << "Unknown S/W Error calling setPreferableTarget() / setPreferableTarget() "
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
    }

    // set input layers and input size for convience later

    try {

        m_InputLayers.resize(1, "input"); 

        if (m_InputOrder == DNNOrder::CH_W_H)
        {
            // ie typically for pytorch
            m_InputShapes[m_InputLayers.at(0)] = MatShape({ batchSize, imageChannels, imageHeight, imageWidth });
            m_InputImageSize = Size(m_InputShapes["input"].at(2),  m_InputShapes["input"].at(3));
            m_InputImageChannels = m_InputShapes["input"].at(1);
        }
        else if (m_InputOrder == DNNOrder::W_H_CH)
        {
            // ie typically for tensor flow (default)
            m_InputLayers.resize(1, "input"); 
            m_InputShapes[m_InputLayers.at(0)] = MatShape({ batchSize, imageHeight, imageWidth, imageChannels });

            m_InputImageSize = Size(m_InputShapes["input"].at(2),  m_InputShapes["input"].at(1));
            m_InputImageChannels = m_InputShapes["input"].at(3);
        }

    } catch(cv::Exception& e) {
		cout << "FATAL ERROR: " 
            << "Opencv Error calling setting input layers: " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(std::exception& e) {
		cout << "FATAL ERROR: " 
            << "S/W Error calling setting input layers: " << e.what() 
            << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(...) {
		cout << "FATAL ERROR: " << "Unknown S/W Error setting input layers in: " <<  __FUNCTION__ << endl;
            exit(1);
	}

    // read output layers ...

    try {

        m_OutputLayers = m_Net.getUnconnectedOutLayersNames();
    
    } catch(cv::Exception& e) {
		cout << "FATAL ERROR: " 
            << "Opencv Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(std::exception& e) {
		cout << "FATAL ERROR: " << "S/W Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	} catch(...) {
		cout << "FATAL ERROR: " << "Unknown S/W Error calling getUnconnectedOutLayersNames(). " 
            << "Check template's image_size and template's onnx_model sizes. " 
            << "In " <<  __FUNCTION__ << endl;
            exit(1);
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
		    cout << "FATAL ERROR: " 
                << "Opencv Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	    } catch(std::exception& e) {
		    cout << "FATAL ERROR: " << "S/W Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	    } catch(...) {
		    cout << "FATAL ERROR: " << "Unknown S/W Error calling getLayerId() / getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << "In " <<  __FUNCTION__ << endl;
            exit(1);
	    }

        // don't think this can ever occur ...

        if (outLayerShapes.size() != 1)
		    cout << "FATAL ERROR: " 
                << "Received an incorrect outLayerShape. Check onnx_model and ensure a valid yolo v3 or v4 model. "
                << "In " <<  __FUNCTION__ << endl;

        try {
           
            m_InputShapes[outputLayer] = outLayerShapes.at(0);
 
        } catch(cv::Exception& e) {
		    cout << "FATAL ERROR: " 
                << "Opencv Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	    } catch(std::exception& e) {
		    cout << "FATAL ERROR: " 
                << "S/W Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << e.what() << ". In " <<  __FUNCTION__ << endl;
            exit(1);
	    } catch(...) {
		    cout << "FATAL ERROR: "
                << "Unknown S/W Error calling getLayerShapes(). " 
                << "Check template's image_size and template's onnx_model sizes. " 
                << "In " <<  __FUNCTION__ << endl;
            exit(1);
	    }
    }
};

void Yolo_v7::forward(const cv::Mat& batch)
{
    this->forward(vector<Mat>(1, batch));
}


void Yolo_v7::forward(const std::vector <cv::Mat>& images)
{
    try {

        m_Predictions_xt.clear();
        m_CorrectedBoxes.clear();
        m_Confidences.clear();
        m_Valid.clear();
        m_ClassProbs.clear();

        if (!m_BatchSize || (images.size() != m_BatchSize))
            return;

        // copy to batch by changing color (BGR2RGB for TF network) into it's ROI
        // then convert the whole batch to float

        Mat blob;
        if (m_InputOrder == DNNOrder::W_H_CH)
        {
            // haven't tested this branch there's no TF V7

            Mat batch = Mat(m_InputImageSize.height * m_BatchSize, m_InputImageSize.width, images.at(0).type());
            for (auto& image : images)
            {
                int i = (int) (&image - &images[0]);
                Rect ROI = Rect(Point(0, (i+0)*image.rows), Point(image.cols, (i+1)*image.rows));
                // swapRB will almost always, if not always be true, so this order optimizes
                // without a copy. if its not always true, maybe try to optimize away

                if (m_SwapRB)
                    cvtColor(image, batch(ROI), cv::COLOR_BGR2RGB);
            }
            batch.convertTo(batch, CV_32F, 1.0f/255.0f);
             
            int inputShape[4] = { 
                m_InputShapes["input"].at(0), 
                m_InputShapes["input"].at(1), 
                m_InputShapes["input"].at(2), 
                m_InputShapes["input"].at(3) 
            };

            blob = Mat(4, inputShape, CV_32F, batch.data);
        }
        else if (m_InputOrder == DNNOrder::CH_W_H)
        {
            blob = blobFromImages(images, 1.0, Size(), Scalar(), m_SwapRB);
            blob.convertTo(blob, CV_32F, 1.0f / 255.0f);
        }

        // and predict !

        { std::lock_guard <std::mutex> guard(*m_Mutex);
            m_Net.setInput(blob);
            m_Net.forward(m_Predictions_cv, m_OutputLayers);
        }

    } catch(cv::Exception& e) {
		cout << "Warning: " << "Opencv Error: " << e.what() << ". In " <<  __FUNCTION__ << endl;
	} catch(std::exception& e) {
		cout << "Warning: " << "S/W Error: " << e.what() << ". In " <<  __FUNCTION__ << endl;
	} catch(...) {
		cout << "Warning: " << "Unknown S/W Error in " <<  __FUNCTION__ << endl;
	}
}

void Yolo_v7::decode()
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
 
            m_Predictions_xt.at(o).resize({
                (unsigned long long) m_InputShapes[output].at(0),
                (unsigned long long) m_InputShapes[output].at(1),
                (unsigned long long) m_InputShapes[output].at(2)
            });

            decodeYoloPredictions(
                m_Predictions_xt.at(o), 
                m_NumClasses,
                boxes_x0y0x1y1.at(o),
                objectnesses.at(o),
                class_probs.at(o)
            );
        }    
 
        //// concatenate as required
        xarray <float>& decodedBoxes = boxes_x0y0x1y1.at(0);
        xarray <float>& decodedObjectnesses = objectnesses.at(0);
        xarray <float>& decodedClassProbs = class_probs.at(0);

        splitAndCorrectBoxes(decodedBoxes, 
            Size(m_InputShapes[m_InputLayers.at(0)][2],  m_InputShapes[m_InputLayers.at(0)][1]), 
            m_CorrectedBoxes);
        splitAndCorrectConfidencesAndClassProbs(decodedObjectnesses, decodedClassProbs, m_NumClasses, m_Confidences, m_ClassProbs);

        for (int i=0; i<m_BatchSize; i++)
            dnn::NMSBoxes(m_CorrectedBoxes.at(i), m_Confidences.at(i), m_nmsConfidence, m_nmsThreshold, m_Valid);
 
    } catch(cv::Exception& e) {
		cout << "Warning: " << "Opencv Error: " << e.what() << ". In " <<  __FUNCTION__ << endl;
	} catch(std::exception& e) {
		cout << "Warning: " << "S/W Error: " << e.what() << ". In " <<  __FUNCTION__ << endl;
	} catch(...) {
		cout << "Warning: " << "Unknown S/W Error in " <<  __FUNCTION__ << endl;
	}
}


std::vector <std::vector <cv::Rect>>& Yolo_v7::boxes()
{
    return m_CorrectedBoxes;
}

std::vector <std::vector <float>>& Yolo_v7::confidences()
{
    return m_Confidences;
}

std::vector <std::vector <std::vector<float>>>& Yolo_v7::classProbs()
{
    return m_ClassProbs;
}

std::vector <int>& Yolo_v7::valid()
{
    return m_Valid;
}
