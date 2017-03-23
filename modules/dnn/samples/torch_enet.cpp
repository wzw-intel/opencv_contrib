/*
Sample of using OpenCV dnn module with Torch ENet model.
*/

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace dnn;

#include <fstream>
#include <iostream>
#include <cstdlib>
#include <sstream>
using namespace std;

const String keys =
        "{help h    || Sample app for loading ENet Torch model. "
                       "The model and class names list can be downloaded here: "
                       "https://www.dropbox.com/sh/dywzk3gyb12hpe5/AAD5YkUa8XgMpHs2gCRgmCVCa }"
        "{model m   || path to Torch .net model file (model_best.net) }"
        "{image i   || path to image file }"
        "{i_blob    | _input | input blob name) }"
        "{o_blob    | l367_Deconvolution | output blob name) }"
        "{c_names c || path to file with classnames for channels (categories.txt) }"
        "{result r  || path to save output blob (optional, binary format, NCHW order) }"
        ;

std::vector<String> readClassNames(const char *filename);

static void colorizeSegmentation(const Mat &score, Mat &segm)
{
    int rows = score.size[1];
    int cols = score.size[2];
    int chns = score.size[0];
    vector<Vec3b> colors(chns);
    RNG rng;
    for( int i = 0; i < chns; i++ )
        colors[i] = Vec3b((uchar)rng.uniform(0, 256), (uchar)rng.uniform(0, 256), (uchar)rng.uniform(0, 256));

    Mat maxCl(rows, cols, CV_8UC1);
    Mat maxVal(rows, cols, CV_32FC1);
    for (int ch = 0; ch < chns; ch++)
    {
        for (int row = 0; row < rows; row++)
        {
            const float *ptrScore = score.ptr<float>(ch, row, 0);
            uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
            float *ptrMaxVal = maxVal.ptr<float>(row);
            for (int col = 0; col < cols; col++)
            {
                if (ptrScore[col] > ptrMaxVal[col])
                {
                    ptrMaxVal[col] = ptrScore[col];
                    ptrMaxCl[col] = ch;
                }
            }
        }
    }

    segm.create(rows, cols, CV_8UC3);
    for (int row = 0; row < rows; row++)
    {
        const uchar *ptrMaxCl = maxCl.ptr<uchar>(row);
        Vec3b *ptrSegm = segm.ptr<Vec3b>(row);
        for (int col = 0; col < cols; col++)
        {
            ptrSegm[col] = colors[ptrMaxCl[col]];
        }
    }
    
}

int main(int argc, char **argv)
{
    CommandLineParser parser(argc, argv, keys);

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    String modelFile = parser.get<String>("model");
    String imageFile = parser.get<String>("image");
    String inBlobName = parser.get<String>("i_blob");
    String outBlobName = parser.get<String>("o_blob");

    if (!parser.check())
    {
        parser.printErrors();
        return 0;
    }

    String classNamesFile = parser.get<String>("c_names");
    String resultFile = parser.get<String>("result");

    //! [Create the importer of TensorFlow model]
    Ptr<dnn::Importer> importer;
    try                                     //Try to import TensorFlow AlexNet model
    {
        importer = dnn::createTorchImporter(modelFile);
    }
    catch (const Exception &err)        //Importer can throw errors, we will catch them
    {
        std::cerr << err.msg << std::endl;
    }
    //! [Create the importer of Caffe model]

    if (!importer)
    {
        std::cerr << "Can't load network by using the mode file: " << std::endl;
        std::cerr << modelFile << std::endl;
        exit(-1);
    }

    //! [Initialize network]
    dnn::Net net;
    importer->populateNet(net);
    importer.release();                     //We don't need importer anymore
    //! [Initialize network]

    //! [Prepare blob]
    Mat img = imread(imageFile), imgf;
    if (img.empty())
    {
        std::cerr << "Can't read image from the file: " << imageFile << std::endl;
        exit(-1);
    }

    Size inputImgSize = Size(512, 512);

    if (inputImgSize != img.size())
        resize(img, img, inputImgSize);       //Resize image to input size

    //! [Set input blob]
    net.setImage(inBlobName, img, true);        //set the network input
    //! [Set input blob]

    TickMeter tm;
    tm.start();

    //! [Make forward pass]
    net.forward();                          //compute output
    //! [Make forward pass]

    tm.stop();

    //! [Gather output]
    Mat result = net.getBlob(outBlobName);   //gather output of "prob" layer

    if (!resultFile.empty()) {
        CV_Assert(result.isContinuous());

        ofstream fout(resultFile.c_str(), ios::out | ios::binary);
        fout.write((char*)result.data, result.total() * sizeof(float));
        fout.close();
    }

    std::cout << "Output blob shape: (" << result.size[0] << " x " << result.size[1] << " x " << result.size[2] << ")\n";
    std::cout << "Inference time, ms: " << tm.getTimeMilli()  << std::endl;

    std::vector<String> classNames;
    if(!classNamesFile.empty()) {
        classNames = readClassNames(classNamesFile.c_str());
        if (classNames.size() > result.size[0])
            classNames = std::vector<String>(classNames.begin() + classNames.size() - result.size[0],
                                             classNames.end());
    }

    /*for(int i_c = 0; i_c < prob.channels(); i_c++) {
        ostringstream convert;
        convert << "Channel #" << i_c;

        if(classNames.size() == prob.channels())
            convert << ": " << classNames[i_c];

        imshow(convert.str().c_str(), prob.getPlane(0, i_c));
    }*/
    Mat segm, show;
    colorizeSegmentation(result, segm);
    addWeighted(img, 0.4, segm, 0.6, 0.0, show);
    imshow("show", show);
    waitKey();

    return 0;
} //main


std::vector<String> readClassNames(const char *filename)
{
    std::vector<String> classNames;

    std::ifstream fp(filename);
    if (!fp.is_open())
    {
        std::cerr << "File with classes labels not found: " << filename << std::endl;
        exit(-1);
    }

    std::string name;
    while (!fp.eof())
    {
        std::getline(fp, name);
        if (name.length())
            classNames.push_back(name);
    }

    fp.close();
    return classNames;
}
