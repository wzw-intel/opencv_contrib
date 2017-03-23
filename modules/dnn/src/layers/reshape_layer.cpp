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
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

static size_t shapeTotal(const std::vector<int>& shape)
{
    size_t i, dims = shape.size(), p = 1;
    for( i = 0; i < dims; i++ ) p *= shape[i];

    return p;
}

static void computeShapeByReshapeMask(const std::vector<int> &srcShape,
                                      const std::vector<int> &maskShape,
                                      Range srcRange,
                                      std::vector<int>& dstShape)
{
    int srcDims = (int)srcShape.size();
    int maskDims = (int)maskShape.size();
    if (srcRange == Range::all())
        srcRange = Range(0, srcDims);
    else
        srcRange.end = srcRange.end == INT_MAX ? srcDims : srcRange.end;

    CV_Assert(0 <= srcRange.start && srcRange.start <= srcRange.end && srcRange.end <= srcDims);
    dstShape.assign(srcDims - srcRange.size() + maskDims, 0);

    std::copy(srcShape.begin(), srcShape.begin() + srcRange.start, dstShape.begin());
    std::copy(srcShape.begin() + srcRange.end, srcShape.begin() + srcDims,
              dstShape.begin() + srcRange.start + maskDims);

    int inferDim = -1;
    for (int i = 0; i < maskDims; i++)
    {
        if (maskShape[i] > 0)
        {
            dstShape[srcRange.start + i] = maskShape[i];
        }
        else if (maskShape[i] == 0)
        {
            if (srcRange.start + i >= srcDims)
                CV_Error(Error::StsBadArg, format("Copy dim[%d] (which has zero size) is out of the source shape bounds", srcRange.start + i));
            dstShape[srcRange.start + i] = srcShape[srcRange.start + i];
        }
        else if (maskShape[i] == -1)
        {
            if (inferDim != -1)
                CV_Error(Error::StsAssert, "Duplicate of inferred dim (which is denoted by -1)");
            inferDim = srcRange.start + i;
            dstShape[inferDim] = 1;
        }
        else
            CV_Error(Error::StsBadArg, "maskShape[i] >= -1");
    }

    size_t srcTotal = shapeTotal(srcShape);
    size_t dstTotal = shapeTotal(dstShape);

    if (inferDim != -1)
    {
        if (srcTotal % dstTotal != 0)
            CV_Error(Error::StsBackTrace, "Can't infer a dim denoted by -1");
        
        dstShape[inferDim] = (int)(srcTotal / dstTotal);
    }
    else
    {
        CV_Assert(srcTotal == dstTotal);
    }
}


class ReshapeLayerImpl : public ReshapeLayer
{
public:
    std::vector<int> outShape;
    bool enableReordering;
    bool channelsReduced;
    bool performReordering;

    ReshapeLayerImpl(const std::vector<int> &newShape_, Range applyingRange_, bool enableReordering_)
    {
        newShapeDesc = newShape_;
        newShapeRange = applyingRange_;
        enableReordering = enableReordering_;
        channelsReduced = performReordering = false;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        outputs.resizeVector(1);

        Mat inp = inputs.getMat(0);
        CV_Assert(inp.isContinuous());
        std::vector<int> shape;
        inp.getShape(shape);
        computeShapeByReshapeMask(shape, newShapeDesc, newShapeRange, outShape);
        channelsReduced = inp.dims > (int)outShape.size() || inp.size[0] > outShape[0];
        performReordering = enableReordering && channelsReduced;

        if(performReordering)
            outputs.create((int)outShape.size(), &outShape[0], inp.type(), 0);
        else
        {
            Mat& out = outputs.getMatRef(0);
            out = out.reshape(1, (int)outShape.size(), &outShape[0]);
        }
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        if( performReordering )
        {
            Mat srcBlob = inputs.getMat(0);
            Mat dstBlob = outputs.getMat(0);

            CV_Assert(srcBlob.total() == dstBlob.total() && srcBlob.data != dstBlob.data);

            const float *srcData = srcBlob.ptr<float>();
            float *dstData = dstBlob.ptr<float>();
            int channels = srcBlob.size[0], height = srcBlob.size[1], width = srcBlob.size[2];

            for( int i_h = 0; i_h < height; i_h++ )
            {
                for( int i_w = 0; i_w < width; i_w++ )
                {
                    for( int i_c = 0; i_c < channels; i_c++ )
                    {
                        int src_i = height*width*i_c + width*i_h + i_w;
                        int dst_i = i_c + channels*width*i_h + channels*i_w;

                        dstData[dst_i] = srcData[src_i];
                    }
                }
            }
        }
    }
};

Ptr<ReshapeLayer> ReshapeLayer::create(const std::vector<int> &newShape,
                                       Range applyingRange,
                                       bool enableReordering)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(newShape, applyingRange, enableReordering));
}

Ptr<ReshapeLayer> ReshapeLayer::create(const LayerParams &params)
{
    int axis = params.get<int>("axis", 0);
    int numAxes = params.get<int>("num_axes", -1);
    bool enableReordering = params.get<bool>("reorder_dims", false);
    CV_Assert(numAxes >= -1);
    Range applyingRange = (numAxes == -1) ? Range(axis, INT_MAX) : Range(axis, axis + numAxes);

    std::vector<int> newShape;
    if (params.has("dim"))
    {
        const DictValue &paramShape = params.get("dim");
        int i, n = paramShape.size();
        newShape.resize(n);
        for (i = 0; i < n; i++)
            newShape[i] = paramShape.get<int>(i);
    }

    Ptr<ReshapeLayer> l(new ReshapeLayerImpl(newShape, applyingRange, enableReordering));
    l->setParamsFrom(params);

    return l;
}

}
}
