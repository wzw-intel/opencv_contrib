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
#include <algorithm>

namespace cv
{
namespace dnn2
{

class MVNLayerImpl : public MVNLayer
{
public:
    enum { LTYPE = CV_32F };

    MVNLayerImpl(const String& _name, bool _normvariance, bool _acrosschannels, double _eps) :
        name_(_name), normvariance(_normvariance), acrosschannels(_acrosschannels), eps(_eps)
    {
        finalized=false;
    }
    virtual ~MVNLayerImpl() {}

    String name_;
    bool normvariance;
    bool acrosschannels;
    double eps;
    bool finalized;

    bool isFinalized() const { return finalized; }
    bool normVariance() const { return normvariance; }
    bool acrossChannels() const { return acrosschannels; }
    double getEps() const { return eps; }

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
        int i, j, k;
        int nchannels = dst.size.p[0], rows = dst.size.p[1], cols = dst.size.p[2];
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        size_t esz = src.elemSize();
        size_t sstep = src.step.p[0]/esz, dstep = dst.step.p[0]/esz;
        size_t sstep1 = src.step.p[1]/esz, dstep1 = dst.step.p[1]/esz;
        double inv_n = 1./nchannels;

        for( i = 0; i < rows; i++ )
            for( j = 0; j < cols; j++ )
            {
                const float* srcptr = srcptr0 + i*sstep1 + j;
                float* dstptr = dstptr0 + i*dstep1 + j;
                double s = 0, s2 = 0;

                for( k = 0; k < nchannels; k++ )
                {
                    float v = srcptr[k*sstep];
                    s += v;
                    s2 += v*v;
                }
                double mean = s * inv_n;
                float meanf = (float)mean;

                if( !normvariance )
                {
                    for( k = 0; k < nchannels; k++ )
                        dstptr[k*dstep] = srcptr[k*sstep] - meanf;
                }
                else
                {
                    double sigma = sqrt(std::max(s2 * inv_n - mean*mean, 0.));
                    float scalef = (float)(1./(sigma + eps));
                    for( k = 0; k < nchannels; k++ )
                        dstptr[k*dstep] = (srcptr[k*sstep] - meanf)*scalef;
                }
            }
    }

    void normalizeInSpace(const Mat& src, Mat& dst)
    {
        int k, nchannels = dst.size.p[0], rows = dst.size.p[1], cols = dst.size.p[2];
        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        size_t esz = src.elemSize();
        size_t sstep = src.step.p[0]/esz, dstep = dst.step.p[0]/esz;

        Scalar m, dev;
        for( k = 0; k < nchannels; k++ )
        {
            Mat src_k(rows, cols, LTYPE, (float*)(srcptr0 + k*sstep));
            Mat dst_k(rows, cols, LTYPE, dstptr0 + k*dstep);

            if( normvariance )
            {
                meanStdDev(src_k, m, dev);
                double a = 1./(dev[0] + eps);
                double b = -m[0]*a;

                src_k.convertTo(dst_k, LTYPE, a, b);
            }
            else
            {
                m = mean(src_k);
                subtract(src_k, m, dst_k);
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
        
        if( acrosschannels )
            normalizeAcrossChannels(src, dst);
        else
            normalizeInSpace(src, dst);
    }
};

Ptr<MVNLayer> MVNLayer::create(Net& net, const String& name0, const LayerPin& input,
                               bool normVariance, bool acrossChannels, double eps)
{
    String name = net->suggestLayerName(name0, "mvn");
    Ptr<MVNLayer> layer = makePtr<MVNLayerImpl>(name, normVariance, acrossChannels, eps);
    net->addLayer(layer, input);
    return layer;
}

}
}
