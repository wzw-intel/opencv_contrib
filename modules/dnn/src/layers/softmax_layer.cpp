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
#include <opencv2/core/ocl.hpp>
#include "modules/dnn/opencl_kernels_dnn.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{

class SoftmaxLayerImpl : public SoftmaxLayer
{
public:
    int axis, channels;
    Mat buf;
    size_t outerSize, innerSize;

    SoftmaxLayerImpl(int axis_ = 1) { axis = axis_; }
    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == (size_t)1);
        Mat inp = inputs.getMat(0);
        std::vector<int> shape;
        inp.getShape(shape);

        int i, dims = inp.dims;
        outerSize = innerSize = 1;
        for( i = 0; i < axis; i++ )
            outerSize *= shape[i];
        for( i = axis+1; i < dims; i++ )
            innerSize *= shape[i];
        channels = shape[axis];

        outputs.resizeVector(1);
        outputs.create(dims, &shape[0], inp.type(), 0);
        
        shape[axis] = 1;
        buf.create(dims, &shape[0], inp.type());
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        CV_Assert(src.type() == CV_32F);

        float *srcPtr = src.ptr<float>();
        float *dstPtr = dst.ptr<float>();
        float *bufPtr = buf.ptr<float>();

        CV_Assert(src.isContinuous() && buf.isContinuous());

        size_t outerStep = src.total()/outerSize;
        size_t cnStep = innerSize;

        //compute max along axis
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

            for (int cnDim = 1; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] = std::max(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
            }
        }

        //subtract max
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] = srcPtr[srcOffset + cnDim * cnStep + i] - bufPtr[bufOffset + i];
            }
        }

        cv::exp(dst, dst);

        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            //sum exp along axis
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = 0.f;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] += dstPtr[srcOffset + cnDim * cnStep + i];
            }
            
            //divide by computed sum
            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[srcOffset + cnDim * cnStep + i] /= bufPtr[bufOffset + i];
            }
        }
    }
};

Ptr<SoftmaxLayer> SoftmaxLayer::create(int axis)
{
    return Ptr<SoftmaxLayer>(new SoftmaxLayerImpl(axis));
}

Ptr<SoftmaxLayer> SoftmaxLayer::create(const LayerParams &params)
{
    int axis = params.get<int>("axis", 1);
    Ptr<SoftmaxLayer> l(new SoftmaxLayerImpl(axis));

    l->setParamsFrom(params);
    return l;
}

}
}
