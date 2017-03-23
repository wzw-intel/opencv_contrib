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
#include "modules/dnn/opencl_kernels_dnn.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <algorithm>

namespace cv
{
namespace dnn
{

class LRNLayerImpl : public LRNLayer
{
public:
    Mat buf;

    LRNLayerImpl(int type_ = CHANNEL_NRM, int size_ = 5, double alpha_ = 1,
                 double beta_ = 0.75, double bias_ = 1,
                 bool normBySize_ = true)
    {
        type = type_;
        size = size_;
        alpha = alpha_;
        beta = beta_;
        bias = bias_;
        normBySize = normBySize_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert( inputs.total() == 1 );
        Mat inp = inputs.getMat(0);
        CV_Assert( inp.dims == 3 );
        CV_Assert( type == CHANNEL_NRM || type == SPATIAL_NRM );

        if (type == SPATIAL_NRM)
            buf.create(inp.size[1], inp.size[2], inp.type());

        outputs.resizeVector(1);
        outputs.create(inp.dims, inp.size.p, inp.type(), 0);
    }
    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);

        switch (type)
        {
            case CHANNEL_NRM:
                channelNormalization(src, dst);
                break;
            case SPATIAL_NRM:
                spatialNormalization(src, dst);
                break;
            default:
                CV_Error(Error::StsNotImplemented, "Unimplemented mode of LRN layer");
                break;
        }
    }
    void channelNormalization(const Mat &src, Mat &dst){
        int cn, channels = src.size[0];
        int rows = src.size[1];
        int cols = src.size[2];
        int ksize = (size - 1) / 2;
        float a = normBySize ? alpha/size : alpha;
        float b = bias;
        float p = -beta;
        size_t esz = src.elemSize();
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        size_t sstep0 = src.step[0]/esz;
        size_t dstep0 = dst.step[0]/esz;
        size_t sstep1 = src.step[1]/esz;
        size_t dstep1 = dst.step[1]/esz;

        for( int i = 0; i < rows; i++ )
        {
            for( int j = 0; j < cols; j++ )
            {
                const float* sptr = srcptr0 + sstep1*i + j;
                float* dptr = dstptr0 + dstep1*i + j;

                float accum = 0;
                for (cn = 0; cn < std::min(ksize, channels); cn++)
                {
                    float v = sptr[sstep0*cn];
                    accum += v*v;
                }

                for (cn = 0; cn < channels; cn++)
                {
                    if (cn + ksize < channels)
                    {
                        float v = sptr[(cn + ksize)*sstep0];
                        accum += v*v;
                    }

                    if (cn - ksize - 1 >= 0)
                    {
                        float v = sptr[(cn - ksize - 1)*sstep0];
                        accum -= v*v;
                    }

                    float scale = std::pow(accum*a + b, p);
                    dptr[dstep0*cn] = sptr[sstep0*cn]*scale;
                }
            }
        }
    }

    void spatialNormalization(const Mat &src, Mat &dst)
    {
        int cn, channels = src.size[0];
        float a = normBySize ? alpha*(size*size) : alpha;
        float b = bias;
        float p = -beta;
        int rows = src.size[1], cols = src.size[2];

        for (cn = 0; cn < channels; cn++)
        {
            Mat srcp = src.plane(0, cn);
            Mat dstp = dst.plane(0, cn);

            sqrBoxFilter(srcp, dstp, dst.depth(), Size(size, size), Point(-1, -1),
                         false, BORDER_CONSTANT|BORDER_ISOLATED);
            for( int i = 0; i < rows; i++ )
            {
                const float* sptr = srcp.ptr<float>(i);
                float* dptr = dstp.ptr<float>(i);
                
                for( int j = 0; j < cols; j++ )
                {
                    float scale = std::pow(dptr[j]*a + b, p);
                    dptr[j] = sptr[j]*scale;
                }
            }
        }
    }
};

Ptr<LRNLayer> LRNLayer::create(int type, int size, float alpha,
                               float beta, float bias,
                               bool normBySize)
{
    return Ptr<LRNLayer>(new LRNLayerImpl(type, size, alpha, beta, bias, normBySize));
}

Ptr<LRNLayer> LRNLayer::create(const LayerParams& params)
{
    int type = -1;
    String nrmType = params.get<String>("norm_region", "ACROSS_CHANNELS");
    if (nrmType == "ACROSS_CHANNELS")
        type = LRNLayer::CHANNEL_NRM;
    else if (nrmType == "WITHIN_CHANNEL")
        type = LRNLayer::SPATIAL_NRM;
    else
        CV_Error(Error::StsBadArg, "Unknown region type \"" + nrmType + "\"");

    int size = params.get<int>("local_size", 5);
    if (size % 2 != 1 || size <= 0)
        CV_Error(Error::StsBadArg, "LRN layer supports only positive odd values for local_size");

    float alpha = params.get<float>("alpha", 1.f);
    float beta = params.get<float>("beta", 0.75f);
    float bias = params.get<float>("bias", 1.f);
    bool normBySize = params.get<bool>("norm_by_size", true);
    
    Ptr<LRNLayer> l(new LRNLayerImpl(type, size, alpha, beta, bias, normBySize));
    l->setParamsFrom(params);
    return l;
}

}
}
