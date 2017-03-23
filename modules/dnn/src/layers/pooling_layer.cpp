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
#include "layers_common.hpp"
#include "opencl_kernels_dnn.hpp"
#include <float.h>
#include <algorithm>
#include <opencv2/core/ocl.hpp>
using std::max;
using std::min;

namespace cv
{
namespace dnn
{

class PoolingLayerImpl : public PoolingLayer
{
public:
    Size inp, out;

    //TODO: add ceil_mode param
    PoolingLayerImpl(int type_, bool globalPooling_, Size kernel_=Size(1,1),
                     Size stride_=Size(1,1), Size pad_=Size(0,0), const String& padMode_=String(""))
    {
        type = type_;
        globalPooling = globalPooling_;
        type = type_;
        kernel = kernel_;
        pad = pad_;
        stride = stride_;
        padMode = padMode_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t ninputs = inputs.total();
        CV_Assert(ninputs == 1);
        Mat inp = inputs.getMat(0);
        CV_Assert(inp.dims == 3);

        Size inpsz(inp.size[2], inp.size[1]);

        if(globalPooling)
        {
            kernel = inpsz;
        }

        computeOutputShape(inpsz);
        std::vector<Mat>& outp = outputs.getMatVecRef();
        size_t noutputs = type == MAX ? 2 : 1;

        outp.resize(noutputs);
        int outsz[] = { inp.size[0], out.height, out.width };

        for( size_t i = 0; i < noutputs; i++ )
            outp[i].create(3, outsz, inp.type());
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp0 = inputs.getMat(0);
        Mat out0 = outputs.getMat(0);
        Mat out1;
        switch (type)
        {
            case MAX:
                out1 = outputs.getMat(1);
                maxPooling(inp0, out0, out1);
                break;
            case AVE:
                avePooling(inp0, out0);
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Not implemented");
                break;
        }
    }

    void computeOutputShape(Size inpSz)
    {
        if (padMode.empty()) {
            //Yeah, some strange Caffe scheme-)
            out.height = static_cast<int>(ceil(static_cast<float>(inpSz.height + 2 * pad.height -
                                                                  kernel.height) / stride.height)) + 1;
            out.width = static_cast<int>(ceil(static_cast<float>(inpSz.width + 2 * pad.width -
                                                                 kernel.width) / stride.width)) + 1;

            if (pad.height || pad.width)
            {
                // If we have padding, ensure that the last pooling starts strictly
                // inside the image (instead of at the padding); otherwise clip the last.
                if ((out.height - 1) * stride.height >= inpSz.height + pad.height)
                    --out.height;
                if ((out.width - 1) * stride.width >= inpSz.width + pad.width)
                    --out.width;
                CV_Assert((out.height - 1) * stride.height < inpSz.height + pad.height);
                CV_Assert((out.width - 1) * stride.width < inpSz.width + pad.width);
            }
        }
        else
        {
            getConvPoolOutParams(inpSz.height, inpSz.width, kernel, stride, pad,
                                 padMode, out.height, out.width);
        }
    }

    void maxPooling(const Mat &src, Mat &dst, Mat &mask)
    {
        CV_DbgAssert(dst.rows == out.height && dst.cols == out.width);

        for (int c = 0; c < src.channels(); ++c)
        {
            const float *srcData = src.ptr<float>(c);
            float *dstData = dst.ptr<float>(c);
            float *dstMaskData = mask.ptr<float>(c);

            for (int ph = 0; ph < out.height; ++ph)
            {
                for (int pw = 0; pw < out.width; ++pw)
                {
                    int hstart = ph * stride.height - pad.height;
                    int wstart = pw * stride.width - pad.width;
                    int hend = min(hstart + kernel.height, inp.height);
                    int wend = min(wstart + kernel.width, inp.width);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    const int poolIndex = ph * out.width + pw;
                    float max_val = -FLT_MAX;
                    int max_index = -1;

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                        {
                            const int index = h * inp.width + w;
                            if (srcData[index] > max_val)
                            {
                                max_val = srcData[index];
                                max_index = index;
                            }
                        }
                    
                    dstData[poolIndex] = max_val;
                    dstMaskData[poolIndex] = max_index;
                }
            }
        }
    }

    void avePooling(const Mat &src, Mat &dst)
    {
        for (int c = 0; c < src.channels(); ++c)
        {
            const float *srcData = src.ptr<float>(c);
            float *dstData = dst.ptr<float>(c);

            for (int ph = 0; ph < out.height; ++ph)
            {
                for (int pw = 0; pw < out.width; ++pw)
                {
                    int hstart = ph * stride.height - pad.height;
                    int wstart = pw * stride.width - pad.width;
                    int hend = min(hstart + kernel.height, inp.height + pad.height);
                    int wend = min(wstart + kernel.width, inp.width + pad.width);
                    int poolSize = (hend - hstart) * (wend - wstart);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    hend = min(hend, inp.height);
                    wend = min(wend, inp.width);

                    dstData[ph * out.width + pw] = 0.f;

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                            dstData[ph * out.width + pw] += srcData[h * inp.width + w];
                    
                    dstData[ph * out.width + pw] /= poolSize;
                }
            }
        }
    }
};


Ptr<PoolingLayer> PoolingLayer::create(int type, Size kernel, Size stride, Size pad,
                                       const String& padMode)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(type, false, kernel, stride, pad, padMode));
}

Ptr<PoolingLayer> PoolingLayer::createGlobal(int type)
{
    return Ptr<PoolingLayer>(new PoolingLayerImpl(type, true));
}

Ptr<PoolingLayer> PoolingLayer::create(const LayerParams& params)
{
    int type = PoolingLayer::MAX;
    Size kernel, stride, pad;
    bool globalPooling;
    cv::String padMode;

    if (params.has("pool"))
    {
        String pool = params.get<String>("pool").toLowerCase();
        if (pool == "max")
            type = PoolingLayer::MAX;
        else if (pool == "ave")
            type = PoolingLayer::AVE;
        else if (pool == "stochastic")
            type = PoolingLayer::STOCHASTIC;
        else
            CV_Error(Error::StsBadArg, "Unknown pooling type \"" + pool + "\"");
    }

    getPoolingKernelParams(params, kernel.height, kernel.width, globalPooling,
                           pad.height, pad.width, stride.height, stride.width, padMode);

    PoolingLayer* l;
    if (!globalPooling)
        l = new PoolingLayerImpl(type, false, kernel, stride, pad, padMode);
    else
        l = new PoolingLayerImpl(type, true);
    l->setParamsFrom(params);

    return Ptr<PoolingLayer>(l);
}

}
}
