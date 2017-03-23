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

#if 0
namespace cv
{
namespace dnn
{

class EltwiseLayerImpl : public EltwiseLayer
{
    EltwiseOp op;
    std::vector<int> coeffs;
public:
    EltwiseLayerImpl(EltwiseOp op, const std::vector<int> &coeffs);
    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs);
    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs);
};

EltwiseLayerImpl::EltwiseLayerImpl(EltwiseOp op_, const std::vector<int> &coeffs_)
{
    op = op_;
    coeffs = coeffs_;
}

void EltwiseLayerImpl::allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
    size_t ninputs = inputs.total(), ncoeffs = coeffs.size();
    CV_Assert(2 <= ninputs);
    CV_Assert(ncoeffs == 0 || ncoeffs == ninputs);
    CV_Assert(op == SUM || ncoeffs == 0);

    Mat inp0 = inputs.getMat(0);
    for (size_t i = 1; i < ninputs; ++i)
    {
        Mat inp = inputs.getMat(i);
        CV_Assert(inp0.size == inp.size);
    }
    std::vector<Mat>& outp = outputs.getMatVecRef();
    outp.resize(1);
    outp[0].create(inp0.dims, inp0.size.p, inp0.type());
}

void EltwiseLayerImpl::forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
{
    size_t i, ninputs = inputs.total(), ncoeffs = coeffs.size();
    Mat output = outputs.getMat(0);

    switch (op)
    {
    case SUM:
        {
            CV_Assert(ncoeffs == 0 || ncoeffs == ninputs);
            output.setTo(0.);
            if (0 < ncoeffs)
            {
                addWeighted(inputs.getMat(0), coeffs[0],
                            inputs.getMat(1), coeffs[1],
                            0, output);
                for (i = 2; i < ninputs; i++)
                {
                    Mat inp = inputs.getMat(i);
                    scaleAdd(inp, coeffs[i], output, output);
                }
            }
            else
            {
                add(inputs.getMat(0), inputs.getMat(1), output);
                for (i = 2; i < ninputs; i++)
                {
                    Mat inp = inputs.getMat(i);
                    add(inp, output, output);
                }
            }
        }
        break;
    case PROD:
        multiply(inputs.getMat(0), inputs.getMat(1), output);
        for (i = 1; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            multiply(inp, output, output);
        }
        break;
    case MAX:
        cv::max(inputs.getMat(0), inputs.getMat(1), output);
        for (i = 1; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            cv::max(inp, output, output);
        }
        break;
    default:
        CV_Assert(0);
        break;
    };
}

Ptr<EltwiseLayer> EltwiseLayer::create(EltwiseOp op, const std::vector<int> &coeffs)
{
    return Ptr<EltwiseLayer>(new EltwiseLayerImpl(op, coeffs));
}

}
}
#endif

