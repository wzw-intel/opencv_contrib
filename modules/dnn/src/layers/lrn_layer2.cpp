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
#include "opencv2/core/hal/hal.hpp"
#include <algorithm>
#include <stdlib.h>

namespace cv
{
namespace dnn2
{

class LRNLayerImpl : public LRNLayer
{
public:
    enum { LTYPE = CV_32F };

    LRNLayerImpl(const String& _name, int _type, int _size,
                 float _alpha, float _beta, float _bias) :
        name_(_name), normtype(_type), ksize(_size), alpha(_alpha), beta(_beta), bias(_bias)
    {
        finalized=false;
    }
    virtual ~LRNLayerImpl() {}

    String name_;
    int normtype;
    int ksize;
    float alpha, beta, bias;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getNormType() const { return normtype; }
    int getSize() const { return ksize; }
    float getAlpha() const { return alpha; }
    float getBeta() const { return beta; }
    float getBias() const { return bias; }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_REDUCE; }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1 &&
                  inputSizes[0].size() == 3);
        outputSizes = inputSizes;
        outIdx.resize(1, inplaceMask[0] ? 0 : -1);

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void normalizeAcrossChannels(const Mat& src, Mat& dst)
    {
        int i, j, k, ksz2 = ksize/2;
        size_t esz = src.elemSize();
        int nchannels = dst.size.p[0], rows = dst.size.p[1], cols = dst.size.p[2];
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        size_t sstep = src.step.p[0]/esz, dstep = dst.step.p[0]/esz;
        size_t sstep1 = src.step.p[1]/esz, dstep1 = dst.step.p[1]/esz;
        AutoBuffer<float> _buf(nchannels+ksz2*2+2);
        float* buf = (float*)_buf + ksz2 + 1;
        float a = alpha/ksize;
        float b = bias;
        float p = -beta;

        for( k = 0; k <= ksz2; k++ )
            buf[-k-1] = buf[nchannels + k] = 0.f;

        for( i = 0; i < rows; i++ )
            for( j = 0; j < cols; j++ )
            {
                const float* srcptr = srcptr0 + i*sstep1 + j;
                float* dstptr = dstptr0 + i*dstep1 + j;

                for( k = 0; k < nchannels; k++ )
                {
                    float v = srcptr[k*sstep];
                    buf[k] = v*v;
                }

                double s = 0;
                for( k = 0; k < ksz2; k++ )
                    s += buf[k];

                for( k = 0; k < nchannels; k++ )
                {
                    s += buf[k + ksz2] - buf[k - ksz2 - 1];
                    float v = std::pow((float)s*a + b, p);
                    dstptr[k*dstep] = srcptr[k*sstep]*v;
                }
            }
    }

    void normalizeInSpace(const Mat& src, Mat& dst)
    {
        int i, j, k, ch, ksz2 = ksize/2;
        size_t esz = src.elemSize();
        int nchannels = dst.size.p[0], rows = dst.size.p[1], cols = dst.size.p[2];
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        size_t sstep = src.step.p[0]/esz, dstep = dst.step.p[0]/esz;
        size_t sstep1 = src.step.p[1]/esz, dstep1 = dst.step.p[1]/esz;
        AutoBuffer<double> _buf((cols + ksz2*2 + 2)*3);
        double* buf = (double*)_buf + ksz2 + 1;
        float* zero = (float*)(buf + cols + ksz2 + 1);
        float* dstbuf = zero + cols;
        float a = alpha/(ksize*ksize);
        float b = bias;
        float p = -beta;

        for( k = 0; k <= ksz2; k++ )
            buf[-k-1] = buf[k+cols] = 0.;
        for( k = 0; k < cols; k++ )
            zero[k] = 0.f;

        for( ch = 0; ch < nchannels; ch++ )
        {
            const float* srcptr = srcptr0 + ch*sstep;
            float* dstptr = dstptr0 + ch*dstep;

            for( j = 0; j < cols; j++ )
            {
                double s = 0;
                for( i = 0; i < ksz2; i++ )
                {
                    double v = srcptr[i*sstep1 + j];
                    s += v*v;
                }
                buf[j] = s;
            }

            for( i = 0; i < rows; i++, srcptr += sstep1, dstptr += dstep1 )
            {
                const float* mrow = i <= ksz2 ? zero : srcptr - (ksz2 + 1)*sstep1;
                const float* prow = i + ksz2 >= rows ? zero : srcptr + ksz2*sstep1;

                for( j = 0; j < cols; j++ )
                {
                    double mv = mrow[j], pv = prow[j];
                    buf[j] += (pv - mv)*(pv + mv);
                }

                double s = 0;
                for( j = 0; j < ksz2; j++ )
                    s += buf[j];

                for( j = 0; j < cols; j++ )
                {
                    s += buf[j + ksz2] - buf[j - ksz2 - 1];
                    dstbuf[j] = (float)s*a + b;
                }

                if( p == -1.f )
                    for( j = 0; j < cols; j++ )
                        dstptr[j] = srcptr[j]/dstbuf[j];
                else if( p == -0.5f )
                    for( j = 0; j < cols; j++ )
                    {
                        float v = dstbuf[j];
                        dstptr[j] = srcptr[j]*(sqrt(v)*(1.f/v));
                    }
                else if( p == -0.75f )
                    for( j = 0; j < cols; j++ )
                    {
                        float v = dstbuf[j];
                        dstptr[j] = srcptr[j]*(sqrt(sqrt(v))*(1.f/v));
                    }
                else
                {
                    hal::log32f(dstbuf, dstbuf, cols);
                    for( j = 0; j < cols; j++ )
                        dstbuf[j] *= p;
                    hal::exp32f(dstbuf, dstbuf, cols);
                    for( j = 0; j < cols; j++ )
                        dstptr[j] = srcptr[j]*dstbuf[j];
                }
            }
        }
    }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  src.size == dst.size && src.dims == 3);

        if( normtype == DNN_CHANNEL_NRM )
            normalizeAcrossChannels(src, dst);
        else
            normalizeInSpace(src, dst);
    }
};

Ptr<LRNLayer> LRNLayer::create(Net& net, const String& name0, const LayerPin& input,
                               int normtype, int size, float alpha, float beta, float bias)
{
    String name = net->suggestLayerName(name0, "lrn");
    Ptr<LRNLayer> layer = makePtr<LRNLayerImpl>(name, normtype, size, alpha, beta, bias);
    net->addLayer(layer, input);
    return layer;
}

}
}
