// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Batch Normalization layer.
*/
#include "../precomp.hpp"

namespace cv
{
namespace dnn
{

class MaxUnpoolLayerImpl : public MaxUnpoolLayer
{
public:
    MaxUnpoolLayerImpl(Size outSize_)
    {
        outSize = outSize_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs){
        CV_Assert(inputs.total() == 2);
        Mat inp0 = inputs.getMat(0);
        CV_Assert(inp0.dims == 3);

        int sz[] = { inp0.size[0], outSize.height, outSize.width };
        outputs.resizeVector(1);
        outputs.create(3, sz, inp0.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 2);
        Mat input = inputs.getMat(0);
        Mat indices = inputs.getMat(1);
        Mat output = outputs.getMat(0);
        CV_Assert(input.total() == indices.total());
        CV_Assert(input.size[0] == output.size[0]);
        int channels = input.size[0];

        for( int i_c = 0; i_c < channels; i_c++ )
        {
            Mat inPlane = input.plane(0, i_c);
            Mat idxPlane = indices.plane(0, i_c);
            Mat outPlane = output.plane(0, i_c);
            outPlane.setTo(0.);

            CV_Assert(inPlane.isContinuous() && idxPlane.isContinuous() && outPlane.isContinuous());

            size_t i, area = inPlane.total();
            const float* inptr = inPlane.ptr<float>();
            const float* idxptr = idxPlane.ptr<float>();
            float* outptr = outPlane.ptr<float>();

            for( i = 0; i < area; i++ )
            {
                int index = cvRound(idxptr[i]);
                CV_Assert(0 <= index && index < (int)area);
                outptr[index] = inptr[i];
            }
        }
    }

    Size outSize;
};

Ptr<MaxUnpoolLayer> MaxUnpoolLayer::create(Size unpoolSize)
{
    return Ptr<MaxUnpoolLayer>(new MaxUnpoolLayerImpl(unpoolSize));
}

Ptr<MaxUnpoolLayer> MaxUnpoolLayer::create(const LayerParams& params)
{
    Size outSize(params.get<int>("out_w"),
                 params.get<int>("out_h"));
    Ptr<MaxUnpoolLayer> l(new MaxUnpoolLayerImpl(outSize));
    l->setParamsFrom(params);
    return l;
}

}
}
