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

#include "../precomp.hpp"

namespace cv
{
namespace dnn2
{

class ActivationLayerImpl : public ActivationLayer
{
public:
    enum { LTYPE = CV_32F };

    ActivationLayerImpl( const String& _name, int _activf,
                         const vector<float>& _params )
        : name_(_name), activf(_activf)
    {
        size_t nparams = _params.size();
        CV_Assert( (activf == DNN_RELU && nparams <= 1) ||
                   (activf == DNN_POWER && nparams == 3) ||
                   (activf != DNN_RELU && activf != DNN_POWER && nparams == 0) );
        params0.resize(nparams);
        std::copy(_params.begin(), _params.end(), params0.begin());
        finalized=false;
    }

    virtual ~ActivationLayerImpl() {}
    vector<float> params0;
    vector<float> params;

    String name_;
    int activf;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getActivationType() const { return activf; }
    void getParams(vector<float>& _params) const
    {
        _params.clear();
        std::copy(params0.begin(), params0.end(), std::back_inserter(_params));
    }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_ACTIV; }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1 &&
                  inputSizes[0].size() == 3);
        outputSizes = inputSizes;
        outIdx.resize(1, inplaceMask[0] ? 0 : -1);

        size_t i, nparams = params0.size(), nparams0 = 3;
        params.resize(std::max(nparams, nparams0));
        for( i = 0; i < nparams; i++ )
            params[i] = params0[i];
        for( ; i < nparams0; i++ )
            params[i] = 0.f;

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void process(const float* src, float* dst, size_t blksize) const
    {
        switch(activf)
        {
        case DNN_RELU:
            {
            float slope = params[0];
            if( slope == 0.f )
                for( size_t j = 0; j < blksize; j++ )
                    dst[j] = std::max(src[j], 0.f);
            else
                for( size_t j = 0; j < blksize; j++ )
                {
                    float v = src[j];
                    dst[j] = v > 0.f ? v : v*slope;
                }
            }
            break;
        case DNN_TANH:
            for( size_t j = 0; j < blksize; j++ )
                dst[j] = std::tanh(src[j]);
            break;
        case DNN_SIGMOID:
            for( size_t j = 0; j < blksize; j++ )
                dst[j] = 1.f / (1.f + std::exp(-src[j]));
            break;
        case DNN_ABS:
            for( size_t j = 0; j < blksize; j++ )
                dst[j] = std::abs(src[j]);
            break;
        case DNN_BNLL:
            for( size_t j = 0; j < blksize; j++ )
                dst[j] = std::log(1.f + std::exp(-std::abs(src[j])));
            break;
        case DNN_POWER:
            {
            float power = params[0];
            float scale = params[1];
            float shift = params[2];
            for( size_t j = 0; j < blksize; j++ )
                dst[j] = std::pow(shift + scale*src[j], power);
            }
            break;
        default:
            CV_Error(Error::StsError, "unknown activation function");
        }
    }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  src.size == dst.size && src.dims == 3);

        const Mat* arrays[] = { &src, &dst, 0 };
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        size_t blksize = it.size;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            const float* srcptr = (const float*)ptrs[0];
            float* dstptr = (float*)ptrs[1];
            process(srcptr, dstptr, blksize);
        }
    }
};

Ptr<ActivationLayer> ActivationLayer::create(Net& net, const String& name0,
                                             const LayerPin& input, int activf,
                                             const vector<float>& params)
{
    const char* prefix = activf == DNN_RELU ? "relu" :
                         activf == DNN_TANH ? "tanh" :
                         activf == DNN_SIGMOID ? "sigmoid" :
                         activf == DNN_ABS ? "abs" :
                         activf == DNN_BNLL ? "bnll" :
                         activf == DNN_POWER ? "power" : 0;
    if(!prefix)
        CV_Error(Error::StsBadArg, "Unknown activation function");
    String name = net->suggestLayerName(name0, prefix);
    Ptr<ActivationLayer> layer = makePtr<ActivationLayerImpl>(name, activf, params);
    net->addLayer(layer, input);
    return layer;
}

}
}
