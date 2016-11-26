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
#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn2
{

class PoolingLayerImpl : public PoolingLayer
{
public:
    enum { LTYPE = CV_32F };

    PoolingLayerImpl(const String& _name, int _pooltype, bool _isglobal, Size _ksize, Size _stride, Size _pad) :
        name_(_name), pooltype(_pooltype), isglobal(_isglobal), ksize0(_ksize), stride0(_stride), pad0(_pad)
    {
        finalized=false;
    }
    virtual ~PoolingLayerImpl() {}

    String name_;
    int pooltype;
    bool isglobal;
    Size ksize0, stride0, pad0;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getPoolingType() const { return pooltype; }
    Size getKernelSize() const { return ksize0; }
    Size getStride() const { return stride0; }
    Size getPadding() const { return pad0; }
    bool isGlobal() const { return isglobal; }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_POOLING; }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1 &&
                  inputSizes[0].size() == 3);
        outputSizes.resize(1);
        outputSizes[0].resize(3);
        Size outSz = getOutputSize(Size(inputSizes[0][2], inputSizes[0][1]));
        outputSizes[0][0] = inputSizes[0][0];
        outputSizes[0][1] = outSz.height;
        outputSizes[0][2] = outSz.width;
        outIdx.resize(1, -1);

        finalized = true;
    }

    Size getOutputSize(Size inSize) const
    {
        Size outSize;
        if(isglobal)
            return Size(1, 1);
        outSize.height = (inSize.height + 2 * pad0.height - 1) / stride0.height + 1;
        outSize.width = (inSize.width + 2 * pad0.width - 1) / stride0.width + 1;

        if (pad0.height || pad0.width)
        {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((outSize.height - 1) * stride0.height >= inSize.height + pad0.height)
                --outSize.height;
            if ((outSize.width - 1) * stride0.width >= inSize.width + pad0.width)
                --outSize.width;
            CV_Assert((outSize.height - 1) * stride0.height < inSize.height + pad0.height);
            CV_Assert((outSize.width - 1) * stride0.width < inSize.width + pad0.width);
        }
        return outSize;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);

        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  src.dims == 3 && dst.dims == 3);

        int i, j, nchannels = src.size.p[0];
        Size inSize(src.size.p[2], src.size.p[1]);
        Size outSize = getOutputSize(inSize);

        CV_Assert( dst.size.p[0] == src.size.p[0] &&
                   dst.size.p[1] == outSize.height &&
                   dst.size.p[2] == outSize.width );

        size_t esz = src.elemSize();
        size_t sstep = src.step.p[0]/esz, dstep = dst.step.p[0]/esz;
        size_t sstep1 = src.step.p[1]/esz, dstep1 = dst.step.p[1]/esz;
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();

        Size ksize = ksize0, stride = stride0, pad = pad0;
        if(isglobal)
            ksize = Size(1, 1);
        int knum = ksize.width*ksize.height;
        int ptype = pooltype;
        float avgscale0 = 1.f/(knum);

        AutoBuffer<size_t> ofsbuf(knum);
        size_t* ofs = ofsbuf;

        for( i = 0; i < ksize.height; i++ )
            for( j = 0; j < ksize.width; j++ )
                ofs[i*ksize.width + j] = sstep1*i + j;

        int jl_fast = -1, jr_fast = -1;

        if( !isglobal )
        {
            for( j = 0; j < outSize.width; j++ )
            {
                int j0 = j * stride.width - pad.width;
                int j1 = min(j0 + ksize.width, inSize.width);
                j0 = max(j0, 0);
                if( j1 - j0 == ksize.width )
                {
                    jr_fast = j;
                    if(jl_fast < 0)
                        jl_fast = j;
                }
            }
        }

        for( int ch = 0; ch < nchannels; ch++ )
        {
            const float *srcptr = srcptr0 + sstep*ch;
            float *dstptr = dstptr0 + dstep*ch;

            if(isglobal)
            {
                if(ptype == DNN_MAX_POOLING)
                {
                    float mval = -FLT_MAX;
                    for( i = 0; i < inSize.height; i++, srcptr += sstep1)
                        for( j = 0; j < inSize.width; j++ )
                            mval = std::max(mval, srcptr[j]);
                    dstptr[0] = mval;
                }
                else
                {
                    double sval = 0.;
                    for( i = 0; i < inSize.height; i++, srcptr += sstep1)
                        for( j = 0; j < inSize.width; j++ )
                            sval += srcptr[j];
                    dstptr[0] = (float)(sval/(inSize.width*inSize.height));
                }
                continue;
            }

            for( i = 0; i < outSize.height; i++, dstptr += dstep1 )
            {
                int i0 = i * stride.height - pad.height;
                int i1 = min(i0 + ksize.height, inSize.height);
                i0 = max(i0, 0);
                int jl = outSize.width, jr = 0;

                if( i1 - i0 == ksize.height )
                {
                    jl = jl_fast;
                    jr = jr_fast;
                }

                for( j = 0;; )
                {
                    for( ; j < jl; j++ )
                    {
                        int j0 = j * stride.width - pad.width;
                        int j1 = min(j0 + ksize.width, inSize.width);
                        j0 = max(j0, 0);

                        if(ptype == DNN_MAX_POOLING)
                        {
                            float mval = -FLT_MAX;
                            for( int ii = i0; ii < i1; ii++ )
                                for( int jj = j0; jj < j1; jj++ )
                                {
                                    float v = srcptr[ii*sstep1 + jj];
                                    mval = std::max(mval, v);
                                }
                            dstptr[j] = mval;
                        }
                        else
                        {
                            float sval = 0.f;
                            for( int ii = i0; ii < i1; ii++ )
                                for( int jj = j0; jj < j1; jj++ )
                                {
                                    sval += srcptr[ii*sstep1 + jj];
                                }
                            dstptr[j] = sval/((i1 - i0)*(j1 - j0));
                        }
                    }

                    if( j == outSize.width )
                        break;

                    {
                    const float* sptr = srcptr + i0*sstep1 + jl*stride.width - pad.width;
                    if( ptype == DNN_MAX_POOLING )
                    {
                        if( ksize.width == 2 && ksize.height == 2 )
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float mval0 = std::max(sptr[0], sptr[1]);
                                float mval1 = std::max(sptr[sstep1], sptr[sstep1+1]);
                                dstptr[j] = std::max(mval0, mval1);
                            }
                        }
                        else if( ksize.width == 3 && ksize.height == 3 )
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float mval0 = std::max(std::max(sptr[0], sptr[1]), sptr[2]);
                                float mval1 = std::max(std::max(sptr[sstep1], sptr[sstep1+1]), sptr[sstep1+2]);
                                float mval2 = std::max(std::max(sptr[sstep1*2], sptr[sstep1*2+1]), sptr[sstep1*2+2]);
                                dstptr[j] = std::max(std::max(mval0, mval1), mval2);
                            }
                        }
                        else
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float mval = -FLT_MAX;
                                for( int k = 0; k < knum; k++ )
                                {
                                    mval = std::max(mval, sptr[ofs[k]]);
                                }
                                dstptr[j] = mval;
                            }
                        }
                    }
                    else
                    {
                        if( ksize.width == 2 && ksize.height == 2 )
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float sval0 = sptr[0] + sptr[1] + sptr[sstep1] + sptr[sstep1 + 1];
                                dstptr[j] = sval0*0.25f;
                            }
                        }
                        else if( ksize.width == 3 && ksize.height == 3 )
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float sval0 = sptr[0] + sptr[1] + sptr[2];
                                float sval1 = sptr[sstep1] + sptr[sstep1+1] + sptr[sstep1+2];
                                float sval2 = sptr[sstep1*2] + sptr[sstep1*2+1] + sptr[sstep1*2+2];
                                dstptr[j] = (sval0 + sval1 + sval2)*(1.f/9);
                            }
                        }
                        else
                        {
                            for( ; j <= jr; j++, sptr += stride.width )
                            {
                                float sval = 0.f;
                                for( int k = 0; k < knum; k++ )
                                {
                                    sval += sptr[ofs[k]];
                                }
                                dstptr[j] = sval*avgscale0;
                            }
                        }
                    }
                    }

                    jl = outSize.width;
                }
            }
        }
    }
};

Ptr<PoolingLayer> PoolingLayer::create(Net& net, const String& name0, const LayerPin& input,
                               int pooltype, bool isglobal, Size ksize, Size stride, Size pad)
{
    String name = net->suggestLayerName(name0, pooltype == DNN_MAX_POOLING ? "maxpool" : "avgpool");
    Ptr<PoolingLayer> layer = makePtr<PoolingLayerImpl>(name, pooltype, isglobal, ksize, stride, pad);
    net->addLayer(layer, input);
    return layer;
}

}
}
