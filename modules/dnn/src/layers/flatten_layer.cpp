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

class FlattenLayerImpl : public FlattenLayer
{
public:
    int numAxes;
    std::vector<int> resultShape;

    FlattenLayerImpl(int startAxis_, int endAxis_)
    {
        startAxis = startAxis_;
        endAxis = endAxis_;
    }
    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        Mat inp = inputs.getMat(0);
        numAxes = inp.dims;
        CV_Assert(startAxis >= 0);
        CV_Assert(endAxis >= startAxis && endAxis < numAxes);

        size_t flattenedDimensionSize = 1;
        int i;
        for (i = startAxis; i <= endAxis; i++)
        {
            flattenedDimensionSize *= inp.size[i];
        }

        CV_Assert(flattenedDimensionSize == (size_t)(int)flattenedDimensionSize);

        resultShape.clear();
        for (i = 0; i < startAxis; i++)
            resultShape.push_back(inp.size[i]);

        resultShape.push_back((int)flattenedDimensionSize);
        for (i = endAxis + 1; i < numAxes; i++)
            resultShape.push_back(inp.size[i]);

        outputs.resizeVector(1);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, ninputs = inputs.total();
        std::vector<Mat>& outp = outputs.getMatVecRef();
        for (i = 0; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            int newdims = (int)resultShape.size();
            outp[i] = inp.reshape(1, newdims, &resultShape[0]);
        }
    }
};

Ptr<FlattenLayer> FlattenLayer::create(const LayerParams& params)
{
    int startAxis = params.get<int>("axis", 1);
    int endAxis = params.get<int>("end_axis", -1);

    Ptr<FlattenLayer> layer(new FlattenLayerImpl(startAxis, endAxis));
    layer->setParamsFrom(params);
    return layer;
}

Ptr<FlattenLayer> FlattenLayer::create(int startAxis, int endAxis)
{
    return Ptr<FlattenLayer>(new FlattenLayerImpl(startAxis, endAxis));
}

}
}

