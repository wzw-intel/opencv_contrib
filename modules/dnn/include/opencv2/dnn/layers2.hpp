/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef __OPENCV_DNN_DNN_LAYERS2_HPP__
#define __OPENCV_DNN_DNN_LAYERS2_HPP__
#include <opencv2/dnn.hpp>

namespace cv
{
namespace dnn2
{
//! @addtogroup dnn
//! @{

/** @defgroup dnnLayerList Partial List of Implemented Layers
  @{
  This subsection of dnn module contains information about bult-in layers and their descriptions.

  Classes listed here, in fact, provides C++ API for creating intances of bult-in layers.
  In addition to this way of layers instantiation, there is a more common factory API (see @ref dnnLayerFactory), it allows to create layers dynamically (by name) and register new ones.
  You can use both API, but factory API is less convinient for native C++ programming and basically designed for use inside importers (see @ref Importer, @ref createCaffeImporter(), @ref createTorchImporter()).

  Bult-in layers partially reproduce functionality of corresponding Caffe and Torch7 layers.
  In partuclar, the following layers and Caffe @ref Importer were tested to reproduce <a href="http://caffe.berkeleyvision.org/tutorial/layers.html">Caffe</a> functionality:
  - Convolution
  - Deconvolution
  - Pooling
  - InnerProduct
  - TanH, ReLU, Sigmoid, BNLL, Power, AbsVal
  - Softmax
  - Reshape, Flatten, Slice, Split
  - LRN
  - MVN
  - Dropout (since it does nothing on forward pass -))
*/

class CV_EXPORTS_W InputLayer : public BaseLayer
{
public:
    static Ptr<InputLayer> create(Net& net, const String& name);
};

class CV_EXPORTS_W ConvolutionLayer : public BaseLayer
{
public:
    static Ptr<ConvolutionLayer> create(Net& net, const String& name, const LayerPin& input,
                                        int inputChannels, int outputChannels,
                                        Size kernelSize, Size stride = Size(1, 1),
                                        Size pad = Size(0, 0), Size dilation = Size(1, 1));

    virtual int getInputChannels() const = 0;
    virtual int getOutputChannels() const = 0;
    virtual Size getKernelSize() const = 0;
    virtual Size getStride() const = 0;
    virtual Size getPadding() const = 0;
    virtual Size getDilation() const = 0;

    virtual void setWeights(InputArray weights,
                            const Range& irange=Range::all(),
                            const Range& orange=Range::all()) = 0;
    virtual void setBias(InputArray bias,
                         const Range& orange=Range::all()) = 0;
    virtual void getWeights(OutputArray weights,
                            const Range& irange=Range::all(),
                            const Range& orange=Range::all()) const = 0;
    virtual void getBias(OutputArray bias,
                         const Range& orange=Range::all()) const = 0;
};

class CV_EXPORTS_W DeconvolutionLayer : public ConvolutionLayer
{
public:
    static Ptr<DeconvolutionLayer> create(Net& net, const String& name, const LayerPin& input,
                                          int inputChannels, int outputChannels,
                                          Size kernel = Size(3, 3), Size stride = Size(1, 1),
                                          Size pad = Size(0, 0), Size dilation = Size(1, 1));
};

class CV_EXPORTS_W LRNLayer : public BaseLayer
{
public:

    enum Type
    {
        DNN_CHANNEL_NRM,
        DNN_SPATIAL_NRM
    };

    static Ptr<LRNLayer> create(Net& net, const String& name, const LayerPin& input,
                                int type = LRNLayer::DNN_CHANNEL_NRM, int size = 5,
                                float alpha = 1.f, float beta = 0.75f, float bias=1.f);

    virtual int getNormType() const = 0;
    virtual int getSize() const = 0;
    virtual float getAlpha() const = 0;
    virtual float getBeta() const = 0;
    virtual float getBias() const = 0;
};

class CV_EXPORTS_W PoolingLayer : public BaseLayer
{
public:
    enum Type
    {
        DNN_MAX_POOLING,
        DNN_AVG_POOLING
    };

    static Ptr<PoolingLayer> create(Net& net, const String& name, const LayerPin& input,
                                    int ptype = PoolingLayer::DNN_MAX_POOLING, bool isglobal=false,
                                    Size kernel = Size(2, 2),
                                    Size stride = Size(1, 1),
                                    Size pad = Size(0, 0));
    virtual int getPoolingType() const = 0;
    virtual Size getKernelSize() const = 0;
    virtual Size getStride() const = 0;
    virtual Size getPadding() const = 0;
    virtual bool isGlobal() const = 0;
};

class CV_EXPORTS_W SoftmaxLayer : public BaseLayer
{
public:
    static Ptr<SoftmaxLayer> create(Net& net, const String& name,
                                    const LayerPin& input,
                                    int axis = 1);
    virtual int getAxis() const = 0;
};

class CV_EXPORTS_W InnerProductLayer : public BaseLayer
{
public:
    static Ptr<InnerProductLayer> create(Net& net, const String& name,
                                         const LayerPin& input,
                                         int axis = 1);
    virtual int getAxis() const = 0;
};

class CV_EXPORTS_W MVNLayer : public BaseLayer
{
public:
    static Ptr<MVNLayer> create(Net& net, const String& name, const LayerPin& input,
                                bool normVariance = true, bool acrossChannels = false,
                                double eps = 1e-9);
    virtual bool normVariance() const = 0;
    virtual bool acrossChannels() const = 0;
    virtual double getEps() const = 0;
};

class CV_EXPORTS_W ReshapeLayer : public BaseLayer
{
public:
    static Ptr<ReshapeLayer> create(Net& net, const String& name, const LayerPin& input,
                                    const vector<int>& permutation,
                                    const vector<int>& newShape);
    virtual void getPermutation(vector<int>& permutation) const = 0;
    virtual void getNewShape(vector<int>& newshape) const = 0;
};

class CV_EXPORTS_W FlattenLayer : public BaseLayer
{
public:
    static Ptr<FlattenLayer> create(Net& net, const String& name, const LayerPin& input,
                                    const Range& axisRange);
    virtual Range getAxisRange() const = 0;
};

class CV_EXPORTS_W ConcatLayer : public BaseLayer
{
public:
    static Ptr<ConcatLayer> create(Net& net, const String& name,
                                   const vector<LayerPin>& inputs,
                                   int axis);
    static Ptr<ConcatLayer> create(Net& net, const String& name,
                                   const LayerPin& input0,
                                   const LayerPin& input1, int axis);
    virtual int getAxis() const = 0;
};

class CV_EXPORTS_W SliceLayer : public BaseLayer
{
public:
    static Ptr<SliceLayer> create(Net& net, const String& name, const LayerPin& input,
                                  int axis, const vector<int> &sliceIndices=vector<int>());
    virtual int getAxis() const = 0;
    virtual void getSliceIndices(vector<int>& sliceIndices) const = 0;
};

/* Activations */

class CV_EXPORTS_W ActivationLayer : public BaseLayer
{
public:
    enum
    {
        DNN_RELU = 0,
        DNN_TANH = 1,
        DNN_SIGMOID = 2,
        DNN_BNLL = 3,
        DNN_ABS = 4,
        DNN_POWER = 5
    };

    static Ptr<ActivationLayer> create(Net& net, const String& name, const LayerPin& input,
                                       int activationFunc, const vector<float>& params = vector<float>());
    virtual void process(const float* src, float* dst, size_t n) const = 0;
    virtual int getActivationType() const = 0;
    virtual void getParams(vector<float>& params) const = 0;
};

/* Layers using in semantic segmentation */

class CV_EXPORTS_W CropLayer : public BaseLayer
{
public:
    static Ptr<CropLayer> create(Net& net, const String& name,
                                 const LayerPin& input,
                                 const vector<Vec2i>& margins);

    virtual void getMargins(vector<Vec2i>& margins) const = 0;
};

class CV_EXPORTS_W ElemwiseLayer : public BaseLayer
{
public:
    enum
    {
        DNN_PROD = 0,
        DNN_SUM = 1,
        DNN_MAX = 2,
    };

    static Ptr<ElemwiseLayer> create(Net& net, const String& name, const vector<LayerPin>& inputs,
                                     int op, const vector<int> &coeffs=vector<int>());
    static Ptr<ElemwiseLayer> create(Net& net, const String& name,
                                     const LayerPin& input0, const LayerPin& input1,
                                     int op, int coeff0=1, int coeff1=1);

    virtual int getElemwiseOp() const = 0;
    virtual void getCoeffs(vector<int>& coeffs) const = 0;
};

//! @}
//! @}

}
}
#endif
