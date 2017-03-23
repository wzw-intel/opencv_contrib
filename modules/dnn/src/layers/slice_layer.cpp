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
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class SliceLayerImpl : public SliceLayer
{
public:
    SliceLayerImpl(int axis_, const std::vector<int> &sliceIndices_)
    {
        axis = axis_;
        sliceIndices = sliceIndices_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);

        Mat inpBlob = inputs.getMat(0);
        int i, axisSize = inpBlob.size[axis];
        std::vector<Mat>& outp = outputs.getMatVecRef();

        std::vector<int> inpShape;
        inpBlob.getShape(inpShape);

        if (!sliceIndices.empty()) //divide blob with respect to passed parameters
        {
            std::vector<int> outAxisSize;
            int prevSlice = 0;

            for (i = 0; i < (int)sliceIndices.size(); i++)
            {
                if (!(prevSlice < sliceIndices[i] && sliceIndices[i] < axisSize))
                    CV_Error(Error::StsBadArg, "Slice indices should be positive, increased and don't exceed size of sliced dimension");

                outAxisSize.push_back(sliceIndices[i] - prevSlice);
                prevSlice = sliceIndices[i];
            }
            outAxisSize.push_back(axisSize - prevSlice);

            int outdims = outAxisSize.size();
            outp.resize(outdims);
            for (i = 0; i < outdims; i++)
            {
                inpShape[axis] = outAxisSize[i];
                outp[i].create(inpBlob.dims, &inpShape[0], inpBlob.type());
            }
        }
        else //divide blob with respect to count of output blobs
        {
            int noutputs = (int)outp.size();
            CV_Assert(noutputs > 0 && axisSize % noutputs == 0);
            int outAxisSize = axisSize / noutputs;
            
            for (i = 0; i < noutputs; i++)
            {
                inpShape[axis] = outAxisSize;
                outp[i].create(inpBlob.dims, &inpShape[0], inpBlob.type());
            }
        }
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, noutputs = outputs.total();
        Mat inpMat = inputs.getMat(0);
        std::vector<Range> ranges(inpMat.dims, Range::all());

        ranges[axis].start = 0;
        for (i = 0; i < noutputs; i++)
        {
            Mat outp = outputs.getMat(i);
            ranges[axis].end = ranges[axis].start + outp.size[axis];
            inpMat(&ranges[0]).copyTo(outp);
            ranges[axis].start = ranges[axis].end;
        }
    }
};

Ptr<SliceLayer> SliceLayer::create(int axis)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(axis, std::vector<int>()));
}

Ptr<SliceLayer> SliceLayer::create(int axis, const std::vector<int>& sliceIndices)
{
    return Ptr<SliceLayer>(new SliceLayerImpl(axis, sliceIndices));
}

Ptr<SliceLayer> SliceLayer::create(const LayerParams& params)
{
    int axis = params.get<int>("axis", 1);
    std::vector<int> sliceIndices;

    if (params.has("slice_point"))
    {
        const DictValue &indicesValue = params.get("slice_point");
        int i, n = indicesValue.size();
        sliceIndices.resize(n);
        for (i = 0; i < n; i++)
            sliceIndices[i] = indicesValue.get<int>(i);
    }

    Ptr<SliceLayer> l(new SliceLayerImpl(axis, sliceIndices));
    l->setParamsFrom(params);

    return l;
}

}
}
