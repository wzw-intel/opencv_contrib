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
    Size inpsz, outsz;

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

        inpsz = Size(inp.size[2], inp.size[1]);

        if(globalPooling)
        {
            kernel = inpsz;
        }

        outsz = computeOutputShape(inpsz);
        std::vector<Mat>& outp = outputs.getMatVecRef();
        size_t noutputs = type == MAX ? 2 : 1;

        outp.resize(noutputs);
        int sz[] = { inp.size[0], outsz.height, outsz.width };

        for( size_t i = 0; i < noutputs; i++ )
            outp[i].create(3, sz, inp.type());
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp0 = inputs.getMat(0);
        Mat out0 = outputs.getMat(0);
        Mat out1;
        CV_Assert(inp0.dims == 3 && inp0.size[1] == inpsz.height && inp0.size[2] == inpsz.width );
        CV_Assert(out0.dims == 3 && out0.size[0] == inp0.size[0] &&
                  out0.size[1] == outsz.height && out0.size[2] == outsz.width );

        switch (type)
        {
            case MAX:
                out1 = outputs.getMat(1);
                CV_Assert(out1.dims == 3 && out1.size[0] == inp0.size[0] &&
                          out1.size[1] == outsz.height && out1.size[2] == outsz.width );
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

    Size computeOutputShape(Size inpSz)
    {
        Size out;
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
        return out;
    }

    void maxPooling(const Mat &src, Mat &dst, Mat &mask)
    {
        Size isz = inpsz, osz = outsz;
        int nchannels = src.size[0];
        CV_Assert(src.type() == CV_32F && dst.type() == CV_32F && mask.type() == CV_32F);
        CV_Assert(src.dims == 3 && dst.dims == 3 && mask.dims == 3);
        CV_Assert(dst.size[1] == osz.height && dst.size[2] == osz.width);
        CV_Assert(mask.size[1] == osz.height && mask.size[2] == osz.width);
        CV_Assert(src.size[1] == isz.height && src.size[2] == isz.width);
        CV_Assert(dst.size[0] == nchannels && mask.size[0] == nchannels);
        CV_Assert(src.isContinuous() && dst.isContinuous() && mask.isContinuous());

        /*printf("kernel=(%d x %d), padding=(%d x %d), stride=(%d x %d), isz=(%d x %d), osz=(%d x %d), idxdata=%p\n", kernel.width, kernel.height, pad.width, pad.height, stride.width, stride.height, isz.width, isz.height, osz.width, osz.height, mask.ptr<float>());*/

        for (int c = 0; c < nchannels; ++c)
        {
            const float *srcData = src.ptr<float>(c);
            float *dstData = dst.ptr<float>(c);
            float *dstMaskData = mask.ptr<float>(c);

            for (int ph = 0; ph < osz.height; ++ph)
            {
                for (int pw = 0; pw < osz.width; ++pw)
                {
                    int hstart = ph * stride.height - pad.height;
                    int wstart = pw * stride.width - pad.width;
                    int hend = min(hstart + kernel.height, isz.height);
                    int wend = min(wstart + kernel.width, isz.width);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    int poolIndex = ph * osz.width + pw;
                    float max_val = -FLT_MAX;
                    int max_index = -1;

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                        {
                            int index = h * isz.width + w;
                            float val = srcData[index];
                            if (val > max_val)
                            {
                                max_val = val;
                                max_index = index;
                            }
                        }

                    dstData[poolIndex] = max_val;
                    dstMaskData[poolIndex] = max_index;
                }
            }
        }
        /*double minval=0, maxval=0;
        minMaxIdx(mask, &minval, &maxval);
        printf("minval=%g, maxval=%g\n", minval, maxval);*/
    }

    void avePooling(const Mat &src, Mat &dst)
    {
        Size isz = inpsz, osz = outsz;
        int nchannels = src.size[0];
        CV_Assert(src.type() == CV_32F && dst.type() == CV_32F);
        CV_Assert(src.dims == 3 && dst.dims == 3);
        CV_Assert(dst.size[1] == osz.height && dst.size[2] == osz.width);
        CV_Assert(src.size[1] == isz.height && src.size[2] == isz.width);
        CV_Assert(dst.size[0] == nchannels);
        CV_Assert(src.isContinuous() && dst.isContinuous());

        for (int c = 0; c < src.size[0]; ++c)
        {
            const float *srcData = src.ptr<float>(c);
            float *dstData = dst.ptr<float>(c);

            for (int ph = 0; ph < osz.height; ++ph)
            {
                for (int pw = 0; pw < osz.width; ++pw)
                {
                    int hstart = ph * stride.height - pad.height;
                    int wstart = pw * stride.width - pad.width;
                    int hend = min(hstart + kernel.height, isz.height + pad.height);
                    int wend = min(wstart + kernel.width, isz.width + pad.width);
                    int poolSize = (hend - hstart) * (wend - wstart);
                    hstart = max(hstart, 0);
                    wstart = max(wstart, 0);
                    hend = min(hend, isz.height);
                    wend = min(wend, isz.width);
                    float s = 0.f;

                    for (int h = hstart; h < hend; ++h)
                        for (int w = wstart; w < wend; ++w)
                            s += srcData[h * isz.width + w];
                    
                    dstData[ph * osz.width + pw] = s / poolSize;
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
