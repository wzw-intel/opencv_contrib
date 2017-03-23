// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of shift layer, which adds up const values to blob.
*/

#include "../precomp.hpp"
#include "op_blas.hpp"

namespace cv
{
namespace dnn
{

class ShiftLayerImpl : public ShiftLayer
{
public:
    ShiftLayerImpl() {}

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        Mat inp = inputs.getMat(0);
        CV_Assert(inp.dims == 3 && inp.type() == CV_32F);

        const Mat &biasBlob = blobs[0];
        perElementMode = biasBlob.dims == inp.dims;
        if( perElementMode )
        {
            CV_Assert(biasBlob.total() == inp.total());
        }
        else
        {
            CV_Assert(biasBlob.total() == (size_t)inp.size[0]);
        }

        outputs.resizeVector(1);
        Mat& outp = outputs.getMatRef(0);
        outp = inp;
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp = inputs.getMat(0);
        Mat outp = outputs.getMat(0);
        const Mat &biasBlob = blobs[0];

        if(perElementMode)
            add(inp, biasBlob, outp);
        else
        {
            const float* src = inp.ptr<float>();
            const float* bias = biasBlob.ptr<float>();
            float* dst = outp.ptr<float>();

            CV_Assert(inp.isContinuous() && outp.isContinuous() && biasBlob.isContinuous());
            size_t i, channels = inp.size[0];
            size_t j, chsize = inp.total()/channels;

            for( i = 0; i < channels; i++ )
            {
                float b = bias[i];
                for( j = 0; j < chsize; j++, src += chsize, dst += chsize )
                    dst[j] = src[j] + b;
            }
        }
    }

    bool perElementMode;
};

Ptr<ShiftLayer> ShiftLayer::create(const LayerParams &params)
{
    Ptr<ShiftLayer> l(new ShiftLayerImpl);
    l->setParamsFrom(params);
    CV_Assert(l->blobs.size() == (size_t)1);

    return l;
}

}
}
