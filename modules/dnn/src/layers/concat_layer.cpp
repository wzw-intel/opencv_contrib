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

namespace cv
{
namespace dnn
{

class ConcatLayerImpl : public ConcatLayer
{
public:
    ConcatLayerImpl(int axis_ = 1) { axis = axis_-1; }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, ninputs = inputs.total();
        CV_Assert(ninputs > 0);

        Mat inp0 = inputs.getMat(0);
        int k, size[CV_MAX_DIM];
        for( k = 0; k < inp0.dims; k++ )
            size[k] = inp0.size[k];

        for( i = 1; i < ninputs; i++ )
        {
            Mat inp = inputs.getMat(i);
            CV_Assert(inp.dims == inp0.dims);

            for( k = 0; k < inp0.dims; k++ )
            {
                CV_Assert(k == axis || inp.size[k] == inp0.size[k]);
            }
            size[axis] += inp.size[axis];
        }

        outputs.resizeVector(1);
        outputs.create(inp0.dims, size, inp0.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, ninputs = inputs.total();
        Mat outMat = outputs.getMat(0);
        std::vector<Range> ranges(outMat.dims, Range::all());

        ranges[axis].start = 0;
        for (i = 0; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            ranges[axis].end = ranges[axis].start + inp.size[axis];
            inp.copyTo(outMat(&ranges[0]));
            ranges[axis].start = ranges[axis].end;
        }
    }
};

Ptr<ConcatLayer> ConcatLayer::create(int axis)
{
    return Ptr<ConcatLayer>(new ConcatLayerImpl(axis));
}

Ptr<ConcatLayer> ConcatLayer::create(const LayerParams& params)
{
    int axis = params.get<int>("axis", 1);
    Ptr<ConcatLayer> layer(new ConcatLayerImpl(axis));
    layer->setParamsFrom(params);
    return layer;
}

}
}
