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
#include <algorithm>
#include <iterator>

namespace cv
{
namespace dnn2
{

class ElemwiseLayerImpl : public ElemwiseLayer
{
public:
    enum { LTYPE = CV_32F, MAX_INPUTS=16 };

    ElemwiseLayerImpl( const String& _name, int _ninputs, int _elemwiseOp,
                       const vector<int>& _coeffs )
    : name_(_name), ninputs(_ninputs), elemwiseOp(_elemwiseOp)
    {
        CV_Assert(2 <= ninputs && ninputs < MAX_INPUTS);
        size_t ncoeffs = _coeffs.size();
        CV_Assert(ncoeffs == 0 || (elemwiseOp == DNN_SUM && ncoeffs == (size_t)ninputs));

        coeffs0.resize(ncoeffs);
        std::copy(_coeffs.begin(), _coeffs.end(), coeffs0.begin());
        finalized=false;
    }

    virtual ~ElemwiseLayerImpl() {}
    vector<int> coeffs0;
    vector<int> coeffs;

    String name_;
    int ninputs;
    int elemwiseOp;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getElemwiseOp() const { return elemwiseOp; }
    void getCoeffs(vector<int>& _coeffs) const
    {
        _coeffs.clear();
        std::copy(coeffs0.begin(), coeffs0.end(), std::back_inserter(_coeffs));
    }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_ELEMWISE; }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == (size_t)ninputs);
        outputSizes.resize(1, inputSizes[0]);
        outIdx.resize(1);
        outIdx[0] = -1;

        int i, j, dims = (int)inputSizes[0].size();
        CV_Assert( dims > 0 );

        // check that all inputs have the same size
        for( i = 0; i < ninputs; i++ )
        {
            for( j = 0; j < dims; j++ )
                CV_Assert(inputSizes[i][j] == outputSizes[0][j]);
            if( inplaceMask[i] && outIdx[0] < 0 )
                outIdx[0] = i;
        }

        coeffs.resize(ninputs);
        int ncoeffs = (int)coeffs0.size();
        for( i = 0; i < ncoeffs; i++ )
            coeffs[i] = coeffs0[i];
        for( ; i < ninputs; i++ )
            coeffs[i] = 1;

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat dst = outputs.getMat(0);
        CV_Assert(dst.type() == LTYPE && dst.dims == 3);
        CV_Assert(inputs.size().area() == ninputs);

        Mat src[MAX_INPUTS];
        const Mat* arrays[MAX_INPUTS+2] = {0};
        uchar* ptrs[MAX_INPUTS+1] = {0};

        int k, ninp = ninputs;

        for(k = 0; k < ninp; k++)
        {
            src[k] = inputs.getMat(k);
            arrays[k+1] = &src[k];
            CV_Assert(src[k].type() == LTYPE && src[k].size == dst.size);
        }

        arrays[0] = &dst;
        
        NAryMatIterator it(arrays, ptrs);
        size_t j, blksize = it.size;
        const int* cs = &coeffs[0];
        size_t blksize0 = ninp == 2 ? blksize : 256;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            float* dstptr = (float*)ptrs[0];
            const float* srcptr0 = (const float*)ptrs[1];
            const float* srcptr1 = (const float*)ptrs[2];

            switch( elemwiseOp )
            {
            case DNN_PROD:
                for( j = 0; j < blksize; j += blksize0,
                                         dstptr += blksize0,
                                         srcptr0 += blksize0,
                                         srcptr1 += blksize0)
                {
                    size_t dj, blksize1 = std::min(blksize - j, blksize0);
                    for( dj = 0; dj < blksize1; dj++ )
                        dstptr[dj] = srcptr0[dj] * srcptr1[dj];
                    for( k = 2; k < ninp; k++ )
                    {
                        const float* srcptr = (const float*)ptrs[k+1];
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] *= srcptr[dj];
                    }
                }
                break;
            case DNN_SUM:
                for( j = 0; j < blksize; j += blksize0,
                                         dstptr += blksize0,
                                         srcptr0 += blksize0,
                                         srcptr1 += blksize0)
                {
                    size_t dj, blksize1 = std::min(blksize - j, blksize0);
                    int cs0 = cs[0], cs1 = cs[1];
                    if( cs0 == 1 && cs1 == 1 )
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] = srcptr0[dj] + srcptr1[dj];
                    else if( cs0 == 1 && cs1 == -1 )
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] = srcptr0[dj] - srcptr1[dj];
                    else if( cs0 == -1 && cs1 == 1 )
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] = srcptr1[dj] - srcptr0[dj];
                    else
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] = cs0*srcptr0[dj] + cs1*srcptr1[dj];

                    for( k = 2; k < ninp; k++ )
                    {
                        const float* srcptr = (const float*)ptrs[k+1];
                        int csk = cs[k];
                        if( csk == 1 )
                            for( dj = 0; dj < blksize1; dj++ )
                                dstptr[dj] += srcptr[dj];
                        else if( csk == -1 )
                            for( dj = 0; dj < blksize1; dj++ )
                                dstptr[dj] -= srcptr[dj];
                        else
                            for( dj = 0; dj < blksize1; dj++ )
                                dstptr[dj] += csk*srcptr[dj];
                    }
                }
                break;
            case DNN_MAX:
                for( j = 0; j < blksize; j += blksize0,
                                         dstptr += blksize0,
                                         srcptr0 += blksize0,
                                         srcptr1 += blksize0)
                {
                    size_t dj, blksize1 = std::min(blksize - j, blksize0);
                    for( dj = 0; dj < blksize1; dj++ )
                        dstptr[dj] = std::max(srcptr0[dj], srcptr1[dj]);
                    for( k = 2; k < ninp; k++ )
                    {
                        const float* srcptr = (const float*)ptrs[k+1];
                        for( dj = 0; dj < blksize1; dj++ )
                            dstptr[dj] = std::max(dstptr[dj], srcptr[dj]);
                    }
                }
                break;
            default:
                CV_Error(Error::StsBadArg, "Unknown element-wise operation (should be DNN_MUL, DNN_SUM or DNN_MAX)");
            }
        }
    }
};

Ptr<ElemwiseLayer> ElemwiseLayer::create(Net& net, const String& name0,
                                         const vector<LayerPin>& inputs,
                                         int op, const vector<int> &coeffs)
{
    const char* prefix = op == DNN_PROD ? "prod" :
                         op == DNN_MAX ? "max" :
                         op == DNN_SUM && coeffs.size() == 2 &&
                            ((coeffs[0] == 1 && coeffs[0] == -1) ||
                             (coeffs[0] == -1 && coeffs[1] == 1)) ? "diff" :
                         op == DNN_SUM ? "sum" : 0;
    if(!prefix)
        CV_Error(Error::StsBadArg, "Unknown element-wise operation");
    CV_Assert( coeffs.empty() || coeffs.size() == inputs.size() );
    String name = net->suggestLayerName(name0, prefix);
    Ptr<ElemwiseLayer> layer = makePtr<ElemwiseLayerImpl>(name, (int)inputs.size(), op, coeffs);
    net->addLayer(layer, inputs);
    return layer;
}

Ptr<ElemwiseLayer> ElemwiseLayer::create(Net& net, const String& name,
                                         const LayerPin& input0, const LayerPin& input1,
                                         int op, int coeff0, int coeff1)
{
    vector<LayerPin> inputs;
    inputs.push_back(input0);
    inputs.push_back(input1);

    vector<int> coeffs;
    if( op == DNN_PROD || op == DNN_MAX )
    {
        CV_Assert(coeff0 == 1 && coeff1 == 1);
    }
    else
    {
        coeffs.push_back(coeff0);
        coeffs.push_back(coeff1);
    }
    return create(net, name, inputs, op, coeffs);
}

}
}
