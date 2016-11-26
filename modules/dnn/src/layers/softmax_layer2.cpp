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
#include <stdlib.h>

namespace cv
{
namespace dnn2
{

using std::max;

class SoftmaxLayerImpl : public SoftmaxLayer
{
public:
    enum { LTYPE = CV_32F };

    SoftmaxLayerImpl(const String& _name, int _axis) : name_(_name), axis(_axis)
    {
        CV_Assert( 0 <= axis );
        finalized=false;
    }
    virtual ~SoftmaxLayerImpl() {}

    String name_;
    int axis;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getAxis() const { return axis; }

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
        CV_Assert(inputSizes.size() == 1 );
        int dims = (int)inputSizes[0].size();
        CV_Assert((dims == 2 || dims == 3) && axis >= 0 && axis < dims);
        outputSizes = inputSizes;
        outIdx.resize(1, inplaceMask[0] ? 0 : -1);

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  src.size == dst.size && (src.dims == 3 || src.dims == 2));

        const float* srcptr0 = src.ptr<float>();
        float* dstptr0 = dst.ptr<float>();
        int dims = src.dims;
        int i, j, k;
        const int* inputSize = &src.size.p[0];
        int naxis = inputSize[axis];
        int ii = axis == 0 ? 1 : 0;
        int ni = inputSize[ii];
        int jj = axis == ii+1 ? ii+2 : ii+1;
        int nj = jj < dims ? inputSize[jj] : 1;
        jj = jj < dims ? jj : axis;
        size_t sstepi = getStep(src, ii);
        size_t sstepj = getStep(src, jj);
        size_t sstepa = getStep(src, axis);
        AutoBuffer<float> _buf(naxis);
        float* buf = _buf;

        dst.create(dims, &inputSize[0], LTYPE);
        size_t dstepi = getStep(dst, ii);
        size_t dstepj = getStep(dst, jj);
        size_t dstepa = getStep(dst, axis);

        for( i = 0; i < ni; i++ )
        {
            for( j = 0; j < nj; j++ )
            {
                const float* srcptr = srcptr0 + sstepi*i + sstepj*j;
                float* dstptr = dstptr0 + dstepi*i + dstepj*j;
                float maxval = -FLT_MAX;

                for( k = 0; k < naxis; k++ )
                {
                    float val = srcptr[k*sstepa];
                    maxval = std::max(maxval, val);
                    buf[k] = val;
                }

                float sum = 0.f;
                for( k = 0; k < naxis; k++ )
                {
                    float v = std::exp(buf[k] - maxval);
                    sum += v;
                    buf[k] = v;
                }
                sum = sum > 0 ? 1.f/sum : 0;
                for( k = 0; k < naxis; k++ )
                    dstptr[k*dstepa] = buf[k]*sum;
            }
        }
    }
};

Ptr<SoftmaxLayer> SoftmaxLayer::create(Net& net, const String& name0,
                                       const LayerPin& input, int axis)
{
    String name = net->suggestLayerName(name0, "softmax");
    Ptr<SoftmaxLayer> layer = makePtr<SoftmaxLayerImpl>(name, axis);
    net->addLayer(layer, input);
    return layer;
}

}
}
