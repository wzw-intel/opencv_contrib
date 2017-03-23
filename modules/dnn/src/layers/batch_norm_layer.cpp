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

class BatchNormLayerImpl : public BatchNormLayer
{
public:
    BatchNormLayerImpl(float eps_, bool hasWeights_, bool hasBias_) :
     eps(eps_), hasWeights(hasWeights_), hasBias(hasBias_)
    {}

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(blobs.size() == (size_t)4 && inputs.total() == (size_t)1);

        outputs.resizeVector(1);
        Mat inp = inputs.getMat(0);
        outputs.create(inp.dims, inp.size.p, inp.type(), 0);
    }
    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == (size_t)1);
        Mat inpBlob = inputs.getMat(0);
        Mat outBlob = outputs.getMat(0);

        if (hasWeights)
            CV_Assert(inpBlob.size[0] == (int)blobs[2].total());

        if (hasBias)
            CV_Assert(inpBlob.size[0] == (int)blobs[3].total());

        int nchannels = inpBlob.size[0];
        int rows = inpBlob.size[1];
        int cols = inpBlob.size[2];
        int type = inpBlob.type();

        for (int n = 0; n < nchannels; n++)
        {
            float mean = blobs[0].at<float>(n);
            float invstd = 1 / sqrt(blobs[1].at<float>(n) + eps);
            float w = hasWeights ? blobs[2].at<float>(n) : 1;
            float b = hasBias ? blobs[3].at<float>(n) : 0;
            Mat inpBlob_plane(rows, cols, type, inpBlob.ptr<float>(n));
            Mat outBlob_plane(rows, cols, type, outBlob.ptr<float>(n));
            inpBlob_plane.convertTo(outBlob_plane, type, w*invstd, b - mean*(w*invstd));
        }
    }

    float eps;
    bool hasWeights, hasBias;
};

Ptr<BatchNormLayer> BatchNormLayer::create(float eps, bool has_weights, bool has_bias)
{
    return Ptr<BatchNormLayer>(new BatchNormLayerImpl(eps, has_weights, has_bias));
}

Ptr<BatchNormLayer> BatchNormLayer::create(const LayerParams& params)
{
    float eps_ = params.get<float>("eps");
    bool hasWeights_ = params.get<bool>("has_weight", false);
    bool hasBias_ = params.get<bool>("has_bias", false);

    Ptr<BatchNormLayer> layer(new BatchNormLayerImpl(eps_, hasWeights_, hasBias_));
    layer->setParamsFrom(params);
    return layer;
}

}  // namespace dnn
}  // namespace cv
