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
#include "opencv2/core/types_c.h"
#include "layers_common.hpp"
#include <algorithm>
#include <iterator>

namespace cv
{
namespace dnn2
{

class SliceLayerImpl : public SliceLayer
{
public:
    enum { LTYPE = CV_32F, MAX_INPUTS=16 };

    SliceLayerImpl( const String& _name, int _axis, const vector<int>& _sliceIndices )
    : name_(_name), axis(_axis), sliceIndices(_sliceIndices)
    {
        CV_Assert(!sliceIndices.empty());
        finalized=false;
    }

    virtual ~SliceLayerImpl() {}

    String name_;
    int axis;
    vector<int> sliceIndices;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getAxis() const { return axis; }
    void getSliceIndices(vector<int>& _sliceIndices) const
    { _sliceIndices = sliceIndices; }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_GENERIC; }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>&,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1);
        int i, dims = (int)inputSizes[0].size();
        int noutputs = getNumOutputs();
        int left = 0;
        CV_Assert(0 <= axis && axis < dims);

        outputSizes.resize(noutputs, inputSizes[0]);

        for( i = 0; i < noutputs; i++ )
        {
            int right = i < noutputs-1 ? sliceIndices[i] : inputSizes[0][axis];
            CV_Assert( left < right );

            outputSizes[i][axis] = right - left;
            left = right;
        }

        outIdx.resize(noutputs, -1);
        finalized = true;
    }

    int getNumOutputs() const { return (int)sliceIndices.size() + 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        int i, dims = src.dims, noutputs = outputs.size().area();

        CV_Assert(src.type() == LTYPE && (axis >= 0 && axis < dims));
        CV_Assert(outputs.size().area() == noutputs);

        Range idx[CV_MAX_DIM];

        for( i = 0; i < dims; i++ )
            idx[i] = Range::all();

        idx[axis] = Range(0, 0);

        for( i = 0; i < noutputs; i++ )
        {
            Mat dst = outputs.getMat(i);
            idx[axis].end = idx[axis].start + dst.size.p[axis];
            Mat buf(src, idx);
            uchar* data = dst.data;
            
            buf.copyTo(dst);
            CV_Assert(data == dst.data);
            
            idx[axis].start = idx[axis].end;
        }
    }
};

Ptr<SliceLayer> SliceLayer::create(Net& net, const String& name0,
                                   const LayerPin& input, int axis,
                                   const vector<int>& sliceIndices)
{
    String name = net->suggestLayerName(name0, "slice");
    Ptr<SliceLayer> layer = makePtr<SliceLayerImpl>(name, axis, sliceIndices);
    net->addLayer(layer, input);
    return layer;
}

}
}
