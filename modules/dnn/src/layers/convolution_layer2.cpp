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
#include <iostream>

namespace cv
{
namespace dnn2
{

class CV_EXPORTS_W ConvLayerImpl : public DeconvolutionLayer
{
public:
    enum { LTYPE = CV_32F, WTYPE = CV_32F };
    typedef float buftype;

    ConvLayerImpl(const String& _name, bool _deconv,
                  int _inputChannels, int _outputChannels,
                  Size _kernel, Size _stride, Size _pad, Size _dilation)
    : name_(_name), deconv(_deconv),
      inputChannels(_inputChannels), outputChannels(_outputChannels),
      kernelSize(_kernel), stride(_stride), padding(_pad), dilation(_dilation)
    {
        CV_Assert(inputChannels > 0 && outputChannels > 0 && kernelSize.width > 0 && kernelSize.height > 0);
        int wsize[] = { outputChannels, inputChannels, kernelSize.area() };
        weights0.create(3, wsize, WTYPE);
        bias0.create(outputChannels, 1, WTYPE);
        wmask.create(3, wsize, CV_8U);
        bmask.create(outputChannels, 1, CV_8U);
        wmask.setTo(Scalar::all(0));
        bmask.setTo(Scalar::all(0));
        weights0.setTo(Scalar::all(0));
        bias0.setTo(Scalar::all(0));
        setbias = false;

        reset();
    }
    virtual ~ConvLayerImpl() {}

    String name_;
    bool deconv;
    int inputChannels, outputChannels;
    Size kernelSize, stride, padding, dilation;
    bool finalized;
    Mat weights0, bias0;
    Mat wmask, bmask;
    Mat weights, bias;
    bool setbias;
    vector<int> ofsbuf;

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_POOLING; }
    bool isFinalized() const { return finalized; }

    int getInputChannels() const { return inputChannels; }
    int getOutputChannels() const { return outputChannels; }
    Size getKernelSize() const { return kernelSize; }
    Size getStride() const { return stride; }
    Size getPadding() const { return padding; }
    Size getDilation() const { return dilation; }

    void setWeights(InputArray _weights,
                    const Range& _irange,
                    const Range& _orange)
    {
        Range irange = _irange, orange = _orange;
        if( irange == Range::all() ) irange = Range(0, inputChannels);
        if( orange == Range::all() ) orange = Range(0, outputChannels);

        CV_Assert( irange.start >= 0 && irange.end > irange.start && irange.end <= inputChannels );
        CV_Assert( orange.start >= 0 && orange.end > orange.start && orange.end <= outputChannels );
        Mat srcw = _weights.getMat();

        CV_Assert( srcw.dims == 3 && srcw.size.p[0] == orange.end - orange.start &&
                   srcw.size.p[1] == irange.end - irange.start &&
                   srcw.size.p[2] == kernelSize.area() );

        Range wpart[] = { orange, irange, Range::all() };
        Mat dstw(weights0, wpart);
        srcw.convertTo(dstw, WTYPE);
        Mat dstwmask(wmask, wpart);
        dstwmask.setTo(Scalar::all(1));
    }

    void setBias(InputArray _bias, const Range& _orange)
    {
        Range orange = _orange;
        if( orange == Range::all() ) orange = Range(0, outputChannels);

        CV_Assert( orange.start >= 0 && orange.end > orange.start && orange.end <= outputChannels );
        Mat srcb = _bias.getMat();

        CV_Assert( srcb.size() == Size(orange.end - orange.start, 1) );

        Mat dstb = bias0.rowRange(orange);
        srcb.convertTo(dstb, WTYPE);
        Mat dstbmask = bmask.rowRange(orange);
        dstbmask.setTo(Scalar::all(1));
        setbias = true;
    }

    void getWeights(OutputArray _weights,
                    const Range& _irange,
                    const Range& _orange) const
    {
        Range irange = _irange, orange = _orange;
        if( irange == Range::all() ) irange = Range(0, inputChannels);
        if( orange == Range::all() ) orange = Range(0, outputChannels);

        CV_Assert( irange.start >= 0 && irange.end > irange.start && irange.end <= inputChannels );
        CV_Assert( orange.start >= 0 && orange.end > orange.start && orange.end <= outputChannels );

        Range wrange[] = { orange, irange, Range::all() };
        weights0(wrange).copyTo(_weights);
    }

    void getBias(OutputArray _bias, const Range& _orange) const
    {
        Range orange = _orange;
        if( orange == Range::all() ) orange = Range(0, outputChannels);
        CV_Assert( orange.start >= 0 && orange.end > orange.start && orange.end <= outputChannels );
        bias0.rowRange(orange).copyTo(_bias);
    }

    void getOutputSizes(Size inSize, Size& outSize, Size& bufSize) const
    {
        if( !deconv )
        {
            outSize.height = (inSize.height + 2 * padding.height -
                              (dilation.height * (kernelSize.height - 1) +
                               1)) / stride.height + 1;
            outSize.width = (inSize.width + 2 * padding.width -
                             (dilation.width * (kernelSize.width - 1) +
                              1)) / stride.width + 1;
            bufSize = Size(inputChannels*kernelSize.area(), outSize.area());
        }
        else
        {
            outSize.height = stride.height * (inSize.height - 1) +
                            kernelSize.height - 2 * padding.height;
            outSize.width = stride.width * (inSize.width - 1) +
                            kernelSize.width - 2 * padding.width;
            bufSize = Size(outputChannels*kernelSize.area() + inputChannels, inSize.area());
        }
    }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t& _bufsize)
    {
        CV_Assert( inputSizes.size() == 1 &&
                   inputSizes[0].size() == 3 &&
                   inputSizes[0][0] == inputChannels );

        int nz = countNonZero(wmask);
        if( (size_t)nz != wmask.total() )
            CV_Error(Error::StsError, "not all the weights are set!");
        if( setbias )
        {
            int bnz = countNonZero(bmask);
            if( (size_t)bnz != bmask.total() )
                CV_Error(Error::StsError, "some of the bias elements are set, but other are not!");
        }

        outputSizes.resize(1);
        outputSizes[0].resize(3);
        Size outSz, bufSz;
        getOutputSizes(Size(inputSizes[0][2], inputSizes[0][1]), outSz, bufSz);
        outputSizes[0][0] = outputChannels;
        outputSizes[0][1] = outSz.height;
        outputSizes[0][2] = outSz.width;
        outIdx.resize(1, -1);

        int kelems = kernelSize.area(), wcols = inputChannels*kelems;
        Mat w0(outputChannels, wcols, WTYPE, weights0.data);

        if(!deconv)
            w0.copyTo(weights);
        else
            transpose(w0, weights);
        bias0.copyTo(bias);

        _bufsize = (size_t)bufSz.width*bufSz.height;

        ofsbuf.resize(wcols*3);
        for( int i = 0; i < wcols; i++ )
        {
            int plane = i / kelems;
            int ofs = i - plane*kelems;
            int ky = ofs / kernelSize.width;
            int kx = ofs - ky*kernelSize.width;
            ofsbuf[i*3] = plane;
            ofsbuf[i*3+1] = ky*dilation.height;
            ofsbuf[i*3+2] = kx*dilation.width;
        }

        finalized = true;
    }

    void im2row(const Mat& src, Mat& rowbuf)
    {
        Size srcsize = Size(src.size.p[2], src.size.p[1]);
        Size dstsize, bufsize;
        getOutputSizes(srcsize, dstsize, bufsize);

        const float* data_im = src.ptr<float>();
        size_t step_c = src.step.p[0]/sizeof(data_im[0]);
        size_t step = src.step.p[1]/sizeof(data_im[0]);
        buftype* data_row = rowbuf.ptr<buftype>();
        const int* ofsdata = &ofsbuf[0];

        for( int y = 0; y < dstsize.height; y++ )
        {
            int y0 = y * stride.height - padding.height;
            for( int x = 0; x < dstsize.width; x++, data_row += bufsize.width )
            {
                int x0 = x * stride.width - padding.width;
                for( int c = 0; c < bufsize.width; c++ )
                {
                    int plane = ofsdata[c*3];
                    int sy = y0 + ofsdata[c*3+1];
                    int sx = x0 + ofsdata[c*3+2];

                    if( 0 <= sx && sx < srcsize.width && 0 <= sy && sy < srcsize.height )
                        data_row[c] = (buftype)data_im[plane*step_c + sy*step + sx];
                    else
                        data_row[c] = (buftype)0;
                }
            }
        }
    }

    void row2im(const Mat& src, Mat& dst)
    {
        CV_Assert( dst.isContinuous() );

        int width = dst.size.p[2], height = dst.size.p[1];
        int nchannels = src.size.p[0];

        int height_col = (height + 2 * padding.height - (dilation.height * (kernelSize.height - 1) + 1)) / stride.height + 1;
        int width_col = (width + 2 * padding.width - (dilation.width * (kernelSize.width - 1) + 1)) / stride.width + 1;
        int veclength = nchannels * kernelSize.height * kernelSize.width;
        const float* srcptr = src.ptr<float>();
        float* dstptr = dst.ptr<float>();
        size_t step_c = dst.step.p[0]/sizeof(dstptr[0]);
        size_t step = dst.step.p[1]/sizeof(dstptr[0]);
        size_t sstep_c = width_col*height_col;
        const int* ofsdata = &ofsbuf[0];

        dst.setTo(Scalar::all(0));
        for( int y = 0; y < height_col; y++ )
        {
            int y0 = y * stride.height - padding.height;
            for( int x = 0; x < width_col; x++, srcptr++ )
            {
                int x0 = x * stride.width - padding.width;
                for( int c = 0; c < veclength; c++ )
                {
                    int plane = ofsdata[c*3];
                    int sy = y0 + ofsdata[c*3+1];
                    int sx = x0 + ofsdata[c*3+2];

                    if( 0 <= sx && sx < width && 0 <= sy && sy < height )
                        dstptr[plane*step_c + sy*step + sx] += srcptr[sstep_c*plane];
                    else
                        dstptr[plane*step_c + sy*step + sx] = (buftype)0;
                }
            }
        }
    }

    class ParallelGEMM : public ParallelLoopBody
    {
    public:
        ParallelGEMM(const float* _src1,
                     const float* _src2,
                     const float* _bias,
                     float* _dst,
                     int _osize, int _veclength) :
            src1_(_src1), src2_(_src2), bias_(_bias), dst_(_dst),
            osize_(_osize), veclength_(_veclength)
        {
        }

        void operator()(const Range& range) const
        {
            const float* src1 = src1_;
            const float* src2 = src1_;
            const float* bias = bias_;
            float* dst = dst_;
            int osize = osize_;
            int veclength = veclength_;
            int oplane = range.start/osize, oidx = range.start - oplane*osize;
            const float* row1 = src1 + oplane*veclength;
            const float* row2 = src2 + oidx*veclength;
            float b0 = bias ? bias[oplane] : 0.f;

            for( int i = range.start; i < range.end; i++ )
            {
                float s = b0;
                for( int j = 0; j < veclength; j++ )
                    s += row1[j]*row2[j];
                dst[i] = s;
                row2 += veclength;
                if( ++oidx >= osize )
                {
                    row2 = src2;
                    row1 += veclength;
                    b0 = bias ? bias[++oplane] : 0.f;
                    oidx = 0;
                }
            }
        }

        const float* src1_;
        const float* src2_;
        const float* bias_;
        float* dst_;
        int osize_, veclength_;
    };

    void gemmWBias(const Mat& src1, const Mat& src2, const Mat& bias, Mat& dst)
    {
        CV_Assert( src1.isContinuous() && src2.isContinuous() &&
                   (bias.empty() || bias.isContinuous()) &&
                   dst.isContinuous() );
        ParallelGEMM invoker(src1.ptr<float>(), src2.ptr<float>(),
                             bias.empty() ? 0 : bias.ptr<float>(),
                             dst.ptr<float>(), src2.rows, src2.cols);
        double granularity = 10000000./((double)src1.total());
        parallel_for_(Range(0, (int)dst.total()), invoker, granularity);
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray _buf)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);

        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  src.dims == 3 && dst.dims == 3 );
        CV_Assert(src.size.p[0] == inputChannels );

        Size inSize(src.size.p[2], src.size.p[1]);
        Size outSize, bufSize;
        getOutputSizes(inSize, outSize, bufSize);

        CV_Assert( dst.size.p[0] == outputChannels &&
                   dst.size.p[1] == outSize.height &&
                   dst.size.p[2] == outSize.width );
        _buf.fit(bufSize, LTYPE);
        Mat buf = _buf.getMat();

        if(!deconv)
        {
            im2row(src, buf);
            gemmWBias(weights, bias, buf, dst);
        }
        else
        {
            Mat buf0;
            gemmWBias(weights, Mat(), src, buf0);
            row2im(buf0, dst);
        }
    }
};


Ptr<ConvolutionLayer> ConvolutionLayer::create(Net& net, const String& name0, const LayerPin& input,
                                               int inputChannels, int outputChannels,
                                               Size kernel, Size stride, Size pad, Size dilation)
{
    String name = net->suggestLayerName(name0, format("conv%dx%d_", kernel.width, kernel.height));
    Ptr<ConvolutionLayer> layer = makePtr<ConvLayerImpl>(name, false, inputChannels, outputChannels,
                                                         kernel, stride, pad, dilation);
    net->addLayer(layer, input);
    return layer;
}

Ptr<DeconvolutionLayer> DeconvolutionLayer::create(Net& net, const String& name0, const LayerPin& input,
                                                   int inputChannels, int outputChannels,
                                                   Size kernel, Size stride, Size pad, Size dilation)
{
    String name = net->suggestLayerName(name0, format("deconv%dx%d_", kernel.width, kernel.height));
    Ptr<DeconvolutionLayer> layer = makePtr<ConvLayerImpl>(name, true, inputChannels, outputChannels,
                                                         kernel, stride, pad, dilation);
    net->addLayer(layer, input);
    return layer;
}

}
}

/*
ConvolutionLayerImpl::ConvolutionLayerImpl()
{
    tryUseOpenCL = false; //true;
    numOutput = -1;
    group = -1;

    #if HAVE_CBLAS
        if (getBlasThreads() != cv::getThreadNum())
        {
            setBlasThreads(cv::getThreadNum());
        }
    #endif
}

void ConvolutionLayerImpl::init()
{
    CV_Assert(1 <= blobs.size() && blobs.size() <= 2);

    bias = (blobs.size() >= 2);
    numOutput = blobs[0].num();

    CV_Assert(blobs[0].dims() == 4 && blobs[0].cols() == kernel.width && blobs[0].rows() == kernel.height);
    CV_Assert(!bias || blobs[1].total() == (size_t)blobs[0].num());

    //TODO: dilation in OCL mode
    useOpenCL = ocl::useOpenCL() && tryUseOpenCL && dilation == Size(1, 1);
}

void ConvolutionLayerImpl::allocate(const std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    init();

    CV_Assert(inputs.size() > 0);
    const Blob &input = *inputs[0];
    CV_Assert(input.dims() == 4 && (input.type() == CV_32F || input.type() == CV_64F));
    computeInpOutShape(input);

    group = inpCn / blobs[0].channels();
    CV_Assert(inpCn % group == 0 && outCn % group == 0);
    CV_Assert(blobs[0].num() == outCn && blobs[0].channels() == inpCn / group);

    outGroupCn = outCn / group;
    inpGroupCn = inpCn / group;
    ksize = inpGroupCn * kernel.height * kernel.width;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Assert(inputs[i]->type() == input.type());
        CV_Assert(inputs[i]->dims() == 4 && inputs[i]->channels() == input.channels());
        CV_Assert(inputs[i]->rows() == input.rows() && inputs[i]->cols() == input.cols());
    }

    int allocFlags = useOpenCL ? Blob::ALLOC_UMAT : Blob::ALLOC_MAT;

    if (!is1x1())
    {
        colBlob.create(Shape(ksize, outH * outW), input.type(), allocFlags);
    }

    if (bias)
    {
        biasOnesBlob.create(Shape(1, topH * topW), input.type(), allocFlags);
        biasOnesBlob.setTo(1);
    }

    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
    {
        outputs[i].create(Shape(inputs[i]->num(), topCn, topH, topW), input.type(), allocFlags);
    }
}

bool ConvolutionLayerImpl::is1x1() const
{
    return (kernel.height == 1 && kernel.width == 1) &&
           (stride.height == 1 && stride.width == 1) &&
           (dilation.height == 1 && dilation.width == 1);
}

template<typename XMat>
void ConvolutionLayerImpl::forward_(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(outCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size(0);
        XMat inpMat = inputs[ii]->getRefConst<XMat>();
        XMat outMat = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*group*outGroupCn, outH*outW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat colMat, curInp = slice(inpMat, n, _Range(g * inpGroupCn, inpGroupCn));
                im2col(curInp, colMat);

                _Range kerRange(g * outGroupCn, outGroupCn);
                XMat kerMat = weightsMat.rowRange(kerRange);

                _Range outRange((g + n * group) * outGroupCn, outGroupCn);
                XMat dstMat = outMat.rowRange(outRange);

                dnn::gemm(kerMat, colMat, 1, dstMat, 0);

                if (bias)
                {
                    dnn::gemm(biasesMat.rowRange(kerRange), biasOnesBlob.getRefConst<XMat>(), 1, dstMat, 1);
                }
            }
        }
    }
}

void ConvolutionLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if (!useOpenCL)
        forward_<Mat>(inputs, outputs);
    else
        forward_<UMat>(inputs, outputs);
}

void ConvolutionLayerImpl::im2col(const UMat &srcImg, UMat &dstCol)
{
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }
#ifdef HAVE_OPENCL
    CV_Assert(im2col_ocl(srcImg, inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dilation.height, dilation.width, this->colBlob.umatRef()));
    dstCol = this->colBlob.umatRefConst();
#else
    CV_Error(Error::StsInternal, "");
    dstCol = srcImg; //supress warning
#endif
}

void ConvolutionLayerImpl::im2col(const Mat &srcImg, Mat &dstCol)
{
    if (is1x1())
    {
        dstCol = reshaped(srcImg, Shape(ksize, outH*outW));
        return;
    }

    Mat &colMat = colBlob.matRef();
    if (srcImg.type() == CV_32F)
        im2col_CpuPBody<float>::run(srcImg.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dilation.height, dilation.width, colMat.ptr<float>());
    if (srcImg.type() == CV_64F)
        im2col_CpuPBody<double>::run(srcImg.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dilation.height, dilation.width, colMat.ptr<double>());

    dstCol = colMat;
}

void ConvolutionLayerImpl::computeInpOutShape(const Blob &input)
{
    inpH = input.rows();
    inpW = input.cols();
    inpCn = input.channels();

    outH = (inpH + 2 * pad.height - (dilation.height * (kernel.height - 1) + 1)) / stride.height + 1;
    outW = (inpW + 2 * pad.width - (dilation.width * (kernel.width - 1) + 1)) / stride.width + 1;
    outCn = numOutput;

    topH = outH; topW = outW; topCn = outCn;
}

//Deconvolution

DeConvolutionLayerImpl::DeConvolutionLayerImpl()
{

}

void DeConvolutionLayerImpl::computeInpOutShape(const Blob &inpBlob)
{
    outH = inpBlob.rows();
    outW = inpBlob.cols();
    outCn = inpBlob.channels();

    inpH = stride.height * (outH - 1) + kernel.height - 2 * pad.height;
    inpW = stride.width * (outW - 1) + kernel.width - 2 * pad.width;
    inpCn = numOutput;

    topH = inpH; topW = inpW; topCn = inpCn;
}

void DeConvolutionLayerImpl::forward(std::vector<Blob*> &inputs, std::vector<Blob> &outputs)
{
    if (!useOpenCL)
        forward_<Mat>(inputs, outputs);
    else
        forward_<UMat>(inputs, outputs);
}

template<typename XMat>
void DeConvolutionLayerImpl::forward_(std::vector<Blob *> &inputs, std::vector<Blob> &outputs)
{
    XMat weightsMat = reshaped(blobs[0].getRefConst<XMat>(), Shape(outCn, ksize));
    XMat biasesMat  = (bias) ? reshaped(blobs[1].getRefConst<XMat>(), Shape(outCn, 1)) : XMat();

    for (size_t ii = 0; ii < outputs.size(); ii++)
    {
        int numImg = inputs[ii]->size(0);
        XMat convBlob = reshaped(inputs[ii]->getRefConst<XMat>(), Shape(numImg*outCn, outH*outW));
        XMat decnBlob = reshaped(outputs[ii].getRef<XMat>(), Shape(numImg*inpCn, inpH*inpW));

        for (int n = 0; n < numImg; n++)
        {
            for (int g = 0; g < group; g++)
            {
                XMat dstMat = decnBlob.rowRange(_Range((g + n * group) * inpGroupCn, inpGroupCn));
                XMat &colMat = (is1x1()) ? dstMat : colBlob.getRef<XMat>();

                XMat convMat = convBlob.rowRange(_Range((g + n * group) * outGroupCn, outGroupCn));
                XMat wghtMat = weightsMat.rowRange(_Range(g * outGroupCn, outGroupCn));

                dnn::gemm(wghtMat, convMat, 1, colMat, 0, GEMM_1_T);

                if (!is1x1())
                    col2im(colMat, dstMat);

                if (bias)
                {
                    XMat curBiasMat = biasesMat.rowRange(_Range(g * outGroupCn, outGroupCn));
                    dnn::gemm(curBiasMat, biasOnesBlob.getRefConst<XMat>(), 1, dstMat, 1);
                }
            }
        }
    }
}

void DeConvolutionLayerImpl::col2im(const Mat &colMat, Mat &dstImg)
{
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
    if (dstImg.type() == CV_32F)
        col2im_CpuPBody<float>::run(colMat.ptr<float>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<float>());
    if (dstImg.type() == CV_64F)
        col2im_CpuPBody<double>::run(colMat.ptr<double>(), inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg.ptr<double>());
}

void DeConvolutionLayerImpl::col2im(const UMat &colMat, UMat &dstImg)
{
    if (is1x1())
    {
        dstImg = colMat;
        return;
    }
#ifdef HAVE_OPENCL
    CV_Assert(col2im_ocl(colMat, inpGroupCn, inpH, inpW, kernel.height, kernel.width, pad.height, pad.width, stride.height, stride.width, dstImg));
#else
    CV_Error(Error::StsInternal, "");
    dstImg = colMat;
#endif
}

//Initializers

Ptr<BaseConvolutionLayer> ConvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    ConvolutionLayerImpl *l = new ConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return Ptr<BaseConvolutionLayer>(l);
}

Ptr<BaseConvolutionLayer> DeconvolutionLayer::create(Size kernel, Size stride, Size pad, Size dilation)
{
    DeConvolutionLayerImpl *l = new DeConvolutionLayerImpl();
    l->kernel = kernel;
    l->pad = pad;
    l->stride = stride;
    l->dilation = dilation;
    return Ptr<BaseConvolutionLayer>(l);
}

}
}
*/
