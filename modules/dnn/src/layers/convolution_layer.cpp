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
#include <opencv2/core/ocl.hpp>
#include "layers_common.hpp"
#include "op_im2col.hpp"
#include "op_blas.hpp"
#include <iostream>

namespace cv
{
namespace dnn
{

class BaseConvolutionLayerImpl : public ConvolutionLayer
{
public:
    BaseConvolutionLayerImpl()
    {
        inpH = inpW = inpCn = 0;
        outH = outW = outCn = 0;
        ksize = 0;
        colBlobCols = 0;
        bias = false;
#if HAVE_CBLAS
        if (getBlasThreads() != cv::getThreadNum())
            setBlasThreads(cv::getThreadNum());
#endif
    }
    virtual void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);
        init();

        Mat input = inputs.getMat(0);
        int datatype = CV_32F;//input.type();
        CV_Assert(input.dims == 3);

        computeInpOutShape(input);

        if (bias)
        {
            biasOnesBlob.create(1, outH * outW, datatype);
            biasOnesBlob.setTo(1);
        }

        outputs.resizeVector(1);
        int outsz[] = { outCn, outH, outW };
        outputs.create(3, outsz, datatype, 0);
        
        if (!is1x1())
            colBlob.create(ksize, colBlobCols, datatype);
    }

protected:
    virtual void computeInpOutShape(const Mat &inpBlob) = 0;
    void init()
    {
        CV_Assert(blobs.size() >= 1 && blobs.size() <= 2);
        Mat& b0 = blobs[0];
        CV_Assert(b0.dims == 4 && b0.size[3] == kernel.width && b0.size[2] == kernel.height);

        bias = (blobs.size() >= 2);
    }
    bool is1x1() const
    {
        return (kernel.height == 1 && kernel.width == 1) &&
               (stride.height == 1 && stride.width == 1) &&
               (dilation.height == 1 && dilation.width == 1);
    }

    int inpH, inpW, inpCn;
    int outH, outW, outCn;
    int ksize;
    int colBlobCols;
    bool bias;

    Mat colBlob, biasOnesBlob;
};

// Convolution
class ConvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:

    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 1);

        Mat weightsMat = blobs[0].reshape(1, outCn);
        Mat biasesMat  = bias ? blobs[1].reshape(1, outCn) : Mat();

        Mat inpMat = inputs.getMat(0), colMat;
        Mat outMat_ = outputs.getMat(0);
        Mat outMat = outMat_.reshape(1, outCn);

        CV_Assert(inpMat.type() == CV_32F && outMat.type() == CV_32F);
        CV_Assert(outMat.cols == outH*outW);
        im2col(inpMat, colMat);

        dnn::gemm(weightsMat, colMat, 1, outMat, 0);

        // TODO: add bias (if any) during the convolution pass, not as a separate step
        if (bias)
            dnn::gemm(biasesMat, biasOnesBlob, 1, outMat, 1);
    }

protected:
    virtual void computeInpOutShape(const Mat &input)
    {
        inpH = input.size[1];
        inpW = input.size[2];
        inpCn = input.size[0];
        outCn = blobs[0].size[0];

        if (padMode.empty())
        {
            outH = (inpH + 2 * pad.height - (dilation.height * (kernel.height - 1) + 1)) / stride.height + 1;
            outW = (inpW + 2 * pad.width - (dilation.width * (kernel.width - 1) + 1)) / stride.width + 1;
        }
        else
        {
            getConvPoolOutParams(inpH, inpW, kernel, stride, pad, padMode, outH, outW);
        }

        CV_Assert(blobs[0].size[1] == inpCn);

        ksize = inpCn * kernel.height * kernel.width;
        
        colBlobCols = outH * outW;
    }
    void im2col(const Mat &srcImg, Mat &dstCol)
    {
        if (is1x1())
        {
            dstCol = srcImg.reshape(1, ksize);
            CV_Assert(dstCol.cols == outH*outW);
        }
        else
        {
            im2col_CpuPBody<float>::run(srcImg.ptr<float>(), inpCn, inpH, inpW, kernel.height,
                                        kernel.width, pad.height, pad.width, stride.height, stride.width,
                                        dilation.height, dilation.width, outH, outW, colBlob.ptr<float>());
            dstCol = colBlob;
        }
    }
};

// Deconvolution, or transposed convolution
class DeconvolutionLayerImpl : public BaseConvolutionLayerImpl
{
public:
    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        Mat b0 = blobs[0];
        Mat weightsMat = b0.reshape(1, inpCn);
        CV_Assert(weightsMat.cols == ksize);

        Mat biasesMat = bias ? blobs[1].reshape(1, outCn) : Mat();

        Mat convBlob = inputs.getMat(0).reshape(1, inpCn);
        CV_Assert( convBlob.cols == inpH*inpW );
        Mat decnBlob = outputs.getMat(0).reshape(1, outCn);
        CV_Assert( decnBlob.cols == outH*outW );

        Mat &colMat = is1x1() ? decnBlob : colBlob;

        dnn::gemm(weightsMat, convBlob, 1, colMat, 0, GEMM_1_T);

        if (!is1x1())
            col2im(colMat, decnBlob);

        if (bias)
            dnn::gemm(biasesMat, biasOnesBlob, 1, decnBlob, 1);
    }

protected:

    virtual void computeInpOutShape(const Mat &inpBlob)
    {
        outCn = blobs[0].size[1];
        CV_Assert(!bias || blobs[1].total() == (size_t)outCn);

        inpH = inpBlob.size[1];
        inpW = inpBlob.size[2];
        inpCn = inpBlob.size[0];

        outH = stride.height * (inpH - 1) + kernel.height - 2 * pad.height + adjustPad.height;
        outW = stride.width * (inpW - 1) + kernel.width - 2 * pad.width + adjustPad.width;

        ksize = outCn * kernel.height * kernel.width;

        colBlobCols = inpH * inpW;
    }

    void col2im(const Mat &colMat, Mat &dstImg)
    {
        if (is1x1())
        {
            dstImg = colMat;
            return;
        }
        col2im_CpuPBody<float>::run(colMat.ptr<float>(), outCn, outH, outW, kernel.height, kernel.width,
                                    pad.height, pad.width, stride.height, stride.width, dstImg.ptr<float>());
    }
};

//Initializers

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    Ptr<BaseConvolutionLayer> l(new ConvolutionLayerImpl);
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation, Size adjustPad)
{
    Ptr<BaseConvolutionLayer> l(new DeconvolutionLayerImpl);
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    l->adjustPad = adjustPad;

    return l;
}

static void initConvDeconvLayerFromCaffe(Ptr<BaseConvolutionLayer>& l, const LayerParams &params)
{
    l->setParamsFrom(params);
    getConvolutionKernelParams(params, l->kernel.height, l->kernel.width, l->pad.height,
                               l->pad.width, l->stride.height, l->stride.width, l->dilation.height,
                               l->dilation.width, l->padMode);

    bool bias = params.get<bool>("bias_term", true);
    int numOutput = params.get<int>("num_output");
    int group = params.get<int>("group", 1);

    l->adjustPad.height = params.get<int>("adj_h", 0);
    l->adjustPad.width = params.get<int>("adj_w", 0);

    CV_Assert(numOutput % group == 0);
    CV_Assert((bias && l->blobs.size() == 2) || (!bias && l->blobs.size() == 1));
}

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new ConvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);
    return l;
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(const LayerParams &params)
{
    Ptr<BaseConvolutionLayer> l(new DeconvolutionLayerImpl);
    initConvDeconvLayerFromCaffe(l, params);
    return l;
}

}
}
