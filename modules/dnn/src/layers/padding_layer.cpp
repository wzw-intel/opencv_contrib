// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of padding layer, which adds paddings to input blob.
*/

#include "../precomp.hpp"
#include <vector>

namespace cv
{
namespace dnn
{

class PaddingLayerImpl : public PaddingLayer
{
public:
    PaddingLayerImpl(int paddingDim_, int padding_,
                     int inputDims_, int index_,
                     float paddingValue_)
    {
        paddingDim = paddingDim_;
        padding = padding_;
        inputDims = inputDims_;
        index = index_;
        paddingValue = paddingValue_;

        if(paddingDim < 0 || padding < 0)
            CV_Error(cv::Error::StsNotImplemented, "Negative padding and dim aren't supported");
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == (size_t)1);
        outputs.resizeVector(1);
        Mat inp = inputs.getMat(0);
        int dim = getPadDim(inp);
        CV_Assert(dim < inp.dims);

        std::vector<int> shape;
        inp.getShape(shape);

        shape[dim] += padding;
        outputs.create(inp.dims, &shape[0], inp.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp = inputs.getMat(0);
        Mat outp = outputs.getMat(0);
        outp.setTo(paddingValue);
        int dim = getPadDim(inp);

        int actualIndex = index;
        if(index == 0)
            actualIndex = inp.size[dim];

        std::vector<std::pair<Range, Range> > srcDstRanges;
        if(actualIndex > 0)
            srcDstRanges.push_back(std::make_pair(Range(0, actualIndex), Range(0, actualIndex)));
        if(actualIndex < inp.size[dim])
        {
            srcDstRanges.push_back(std::make_pair(Range(actualIndex, inp.size[dim]),
                                                  Range(actualIndex + padding, outp.size[dim])));
        }
        std::vector<Range> srcRanges(inp.dims, Range::all()), dstRanges = srcRanges;

        for(size_t i = 0; i < srcDstRanges.size(); i++)
        {
            if(!srcDstRanges[i].first.empty())
            {
                srcRanges[dim] = srcDstRanges[i].first;
                dstRanges[dim] = srcDstRanges[i].second;
                Mat dst = outp(&dstRanges[0]);
                Mat src = inp(&srcRanges[0]);
                src.copyTo(dst);
            }
        }
    }

    int getPadDim(const Mat& inp) const
    {
        return inputDims > 0 && inp.dims > inputDims ? paddingDim + 1 : paddingDim;
    }

    int paddingDim, padding, inputDims, index;
    float paddingValue;
};

Ptr<PaddingLayer> PaddingLayer::create(int paddingDim, int padding,
                                       int inputDims, int index,
                                       float paddingValue)
{
    return Ptr<PaddingLayer>(new PaddingLayerImpl(paddingDim, padding,
                                                  inputDims, index,
                                                  paddingValue));
}

Ptr<PaddingLayer> PaddingLayer::create(const LayerParams& params)
{
    int paddingDim = params.get<int>("padding_dim");
    int padding = abs(params.get<int>("padding"));
    int inputDims = params.get<int>("input_dims", 0);
    int index = params.get<int>("index", 0);
    float paddingValue = (float)params.get<double>("value", 0.);

    Ptr<PaddingLayer> l(new PaddingLayerImpl(paddingDim, padding,
                                inputDims, index, paddingValue));
    l->setParamsFrom(params);

    return l;
}

}
}
