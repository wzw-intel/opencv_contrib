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
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

class PermuteLayerImpl : public PermuteLayer
{
public:
    std::vector<int> order;

    std::vector<int> oldDimensionSize;
    std::vector<int> newDimensionSize;
    std::vector<size_t> oldStride;
    std::vector<size_t> newStride;
    bool needsPermute;
    int numAxes;

    PermuteLayerImpl(const std::vector<int> &order_)
    {
        if(order_.empty())
        {
            needsPermute = false;
            return;
        }

        if(order_.size() > 4)
        {
            CV_Error(Error::StsBadArg,
                "Too many (> 4) orders of dimensions in Permute layer");
        }

        numAxes = (int)order_.size();
        needsPermute = false;

        for (int i = 0; i < numAxes; i++)
        {
            int currentOrder = order_[i];

            if(std::find(order.begin(), order.end(), currentOrder) != order.end())
            {
                CV_Error(Error::StsBadArg,
                    "Permute layer parameter contains duplicated orders.");
            }
            if (currentOrder != i)
                needsPermute = true;
            order.push_back(currentOrder);
        }
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        outputs.resizeVector(1);

        if(!needsPermute)
            return;

        Mat inp = inputs.getMat(0);
        CV_Assert((int)numAxes == inp.dims);

        oldDimensionSize.resize(numAxes);
        newDimensionSize.resize(numAxes);
        oldStride.resize(numAxes);
        newStride.resize(numAxes);

        for( int i = 0; i < numAxes; i++ )
        {
            int currentOrder = order[i];
            CV_Assert(currentOrder >= 0 && currentOrder < inp.dims);

            oldDimensionSize[i] = inp.size[i];
            newDimensionSize[i] = inp.size[currentOrder];
        }

        oldStride[numAxes - 1] = 1;
        newStride[numAxes - 1] = 1;

        for( int i = numAxes - 2; i >= 0; i-- )
        {
            oldStride[i] = oldStride[i + 1] * oldDimensionSize[i + 1];
            newStride[i] = newStride[i + 1] * newDimensionSize[i + 1];
        }

        outputs.create(inp.dims, &newDimensionSize[0], inp.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp = inputs.getMat(0);

        if(!needsPermute)
        {
            Mat& outp = outputs.getMatRef(0);
            outp = inp;
        }
        else
        {
            Mat outp = outputs.getMat(0);
            const float* srcData = inp.ptr<float>();
            float* dstData = outp.ptr<float>();
            size_t i, count = inp.total();
            int j, numAxes = this->numAxes;
            const size_t* newStride = &this->newStride[0];
            const size_t* oldStride = &this->oldStride[0];
            const int* order = &this->order[0];

            CV_Assert(inp.isContinuous() && outp.isContinuous());

            for (i = 0; i < count; ++i)
            {
                size_t oldPosition = 0;
                size_t newPosition = i;
                
                for (j = 0; j < numAxes; ++j)
                {
                    size_t np = newPosition / newStride[j];
                    oldPosition += np * oldStride[order[j]];
                    newPosition -= np*newStride[j];
                }
                dstData[i] = srcData[oldPosition];
            }
        }
    }
};

Ptr<PermuteLayer> PermuteLayer::create(const std::vector<int> &order)
{
    return Ptr<PermuteLayer>(new PermuteLayerImpl(order));
}

Ptr<PermuteLayer> PermuteLayer::create(const LayerParams& params)
{
    std::vector<int> order;
    if (params.has("order"))
    {
        DictValue paramOrder = params.get("order");
        size_t i, numAxes = paramOrder.size();

        for (i = 0; i < numAxes; i++)
        {
            int currentOrder = paramOrder.get<int>(i);
            order.push_back(currentOrder);
        }
    }

    Ptr<PermuteLayer> l(new PermuteLayerImpl(order));
    l->setParamsFrom(params);

    return l;
}

}
}
