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

namespace cv
{
namespace dnn2
{

class ReshapeLayerImpl : public ReshapeLayer
{
public:
    enum { LTYPE = CV_32F };

    ReshapeLayerImpl(const String& _name, const vector<int>& _permut, const vector<int>& _ns) :
        name_(_name), permutation(_permut), newShape(_ns)
    {
        finalized=false;
    }
    virtual ~ReshapeLayerImpl() {}

    String name_;
    vector<int> permutation;
    vector<int> ipermutation;
    vector<int> newShape;
    bool finalized;

    bool isFinalized() const { return finalized; }
    void getPermutation(vector<int>& _permutation) const { _permutation = permutation; }
    void getNewShape(vector<int>& _newshape) const { _newshape = newShape; }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_GENERIC; }

    bool needReshuffle() const
    {
        if(permutation.empty())
            return false;
        size_t i, n = permutation.size();
        for( i = 0; i < n; i++ )
            if( permutation[i] != (int)i )
                return true;
        return false;
    }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1);
        size_t idims = inputSizes[0].size(), odims = newShape.size();
        CV_Assert(permutation.empty() || permutation.size() == idims);
        CV_Assert(odims > 0 && matSize(newShape) == matSize(inputSizes[0]));
        outputSizes.resize(1, newShape);
        outIdx.resize(1, inplaceMask[0] && !needReshuffle() ? 0 : -1);
        if( !permutation.empty() )
        {
            ipermutation.resize(idims);
            for( size_t i = 0; i < idims; i++ )
            {
                int j = permutation[i];
                CV_Assert( 0 <= j && j < (int)idims );
                ipermutation[j] = (int)i;
            }
        }

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        int k, idims = src.dims, odims = (int)newShape.size();
        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  dst.dims == odims && src.total() == dst.total());
        CV_Assert(src.isContinuous() && dst.isContinuous());

        bool reshuffle = needReshuffle();
        if( reshuffle )
        {
            size_t i, j, total = src.total();
            const int* iP = &ipermutation[0];
            const int* ssize = &src.size.p[0];
            AutoBuffer<size_t> _stepbuf(idims);
            size_t* sstep = _stepbuf;
            size_t esz = src.elemSize();
            size_t bsz = ssize[iP[idims-1]];
            const float* sptr0 = src.ptr<float>();
            float* dptr = dst.ptr<float>();

            for( k = 0; k < idims-1; k++ )
                sstep[k] = src.step.p[k]/esz;
            sstep[idims-1] = 1;
            size_t step0 = sstep[iP[idims-1]];

            for( i = 0; i < total; i += bsz, dptr += bsz )
            {
                size_t ofs = 0;
                size_t idx = i;
                for( k = idims-1; k >= 0; k++ )
                {
                    int n = ssize[iP[k]];
                    size_t idx1 = idx / n;
                    int ofs_k = (int)(idx - idx1 * n);
                    ofs += ofs_k * sstep[iP[k]];
                    idx = idx1;
                }
                const float* sptr = sptr0 + ofs;
                for( j = 0; j < bsz; j++, sptr += step0 )
                    dptr[j] = *sptr;
            }

            // dstptr(i0, i1, ...) = srcptr(i[P[0]], i[P[1]], ...)
            // P[0] = 0, P[1] = 2, P[2] = 1: dstptr(i0, i1, i2) = srcptr(i0, i2, i1)
        }
        else if( dst.data != src.data )
        {
            Mat dstAlias(idims, src.size.p, LTYPE, dst.data);
            src.copyTo(dstAlias);
        }
    }
};

Ptr<ReshapeLayer> ReshapeLayer::create(Net& net, const String& name0, const LayerPin& input,
                         const vector<int>& permutation,
                         const vector<int>& newShape)
{
    String name = net->suggestLayerName(name0, "reshape");
    Ptr<ReshapeLayer> layer = makePtr<ReshapeLayerImpl>(name, permutation, newShape);
    net->addLayer(layer, input);
    return layer;
}
    
}
}
