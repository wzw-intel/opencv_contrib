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

class SplitLayerImpl : public SplitLayer
{
public:
    SplitLayerImpl(int outputsCount_ = -1)
    {
        outputsCount = outputsCount_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        Mat inp = inputs.getMat(0);

        std::vector<Mat>& outp = outputs.getMatVecRef();
        size_t i, noutputs = outputsCount > 0 ? (size_t)outputsCount : outp.size();
        outputs.resizeVector(noutputs);

        for (i = 0; i < noutputs; i++)
            outputs.create(inp.dims, &inp.size.p[0], inp.type(), i);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat inp = inputs.getMat(0);
        size_t i, noutputs = outputs.total();
        for (i = 0; i < noutputs; i++)
        {
            Mat& outp = outputs.getMatRef(i);
            inp.copyTo(outp);
        }
    }
};

Ptr<SplitLayer> SplitLayer::create(int outputsCount)
{
    return Ptr<SplitLayer>(new SplitLayerImpl(outputsCount));
}

Ptr<SplitLayer> SplitLayer::create(const LayerParams& params)
{
    //TODO: maybe "top_count" param is useless because it can be determined by output connections number
    int outputsCount = params.get<int>("top_count", -1);
    
    Ptr<SplitLayer> l(new SplitLayerImpl(outputsCount));
    l->setParamsFrom(params);

    return l;
}

}
}
