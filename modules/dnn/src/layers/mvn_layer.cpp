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

class MVNLayerImpl : public MVNLayer
{
public:
    MVNLayerImpl(bool normVariance_ = true, bool acrossChannels_ = false, float eps_ = 1e-9)
    {
        normVariance = normVariance_;
        acrossChannels = acrossChannels_;
        eps = eps_;
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        outputs.resizeVector(1);
        Mat inp = inputs.getMat(0);
        CV_Assert(!acrossChannels || inp.dims >= 2);
        outputs.create(inp.dims, inp.size.p, inp.type(), 1);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        Mat inpMat_ = inputs.getMat(0);
        Mat outMat_ = outputs.getMat(0);
        size_t total = inpMat_.total();
        int rows = acrossChannels ? 1 : inpMat_.size[0];
        int cols = (int)(total/rows);
        CV_Assert((size_t)rows*cols == total);
        int sz[] = { rows, cols };
        Mat inpMat = inpMat_.reshape(1, 2, sz);
        Mat outMat = outMat_.reshape(1, 2, sz);

        for (int i = 0; i < rows; i++)
        {
            Scalar mean, dev;
            Mat inpRow = inpMat.row(i);
            Mat outRow = outMat.row(i);
            meanStdDev(inpRow, mean, normVariance ? dev : noArray());
            double alpha = normVariance ? 1/(eps + dev[0]) : 1;
            inpRow.convertTo(outRow, outRow.type(), alpha, -mean[0] * alpha);
        }
    }
};

Ptr<MVNLayer> MVNLayer::create(bool normVariance, bool acrossChannels, float eps)
{
    return Ptr<MVNLayer>(new MVNLayerImpl(normVariance, acrossChannels, eps));
}

Ptr<MVNLayer> MVNLayer::create(const LayerParams& params)
{
    bool normVariance = params.get<bool>("normalize_variance", true);
    bool acrossChannels = params.get<bool>("across_channels", false);
    float eps = (float)params.get<double>("eps", 1e-9);

    Ptr<MVNLayer> l(new MVNLayerImpl(normVariance, acrossChannels, eps));
    l->setParamsFrom(params);

    return l;
}

}
}
