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

class ConcatLayerImpl : public ConcatLayer
{
public:
    enum { LTYPE = CV_32F, MAX_INPUTS=16 };

    ConcatLayerImpl( const String& _name, int _ninputs, int _axis )
        : name_(_name), ninputs(_ninputs), axis(_axis)
    {
        CV_Assert(2 <= ninputs && ninputs < MAX_INPUTS);
        finalized=false;
    }

    virtual ~ConcatLayerImpl() {}

    String name_;
    int ninputs;
    int axis;
    bool finalized;

    bool isFinalized() const { return finalized; }
    int getAxis() const { return axis; }

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
        CV_Assert(inputSizes.size() == (size_t)ninputs);
        int i, j, dims = (int)inputSizes[0].size();
        int s = 0;
        CV_Assert(0 <= axis && axis < dims);

        for( i = 0; i < ninputs; i++ )
        {
            CV_Assert( dims == (int)inputSizes[i].size() );
            for( j = 0; j < dims; j++ )
            {
                if( j == axis )
                    s += inputSizes[i][j];
                else
                {
                    CV_Assert(inputSizes[i][j] == inputSizes[i][0]);
                }
            }
        }

        outputSizes.resize(1, inputSizes[0]);
        outputSizes[0][axis] = s;
        outIdx.resize(1);
        outIdx[0] = -1;

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat dst = outputs.getMat(0);
        int i, dims = dst.dims;

        CV_Assert(dst.type() == LTYPE && (axis >= 0 && axis < dims));
        CV_Assert(inputs.size().area() == ninputs);

        Range idx[CV_MAX_DIM];

        for( i = 0; i < dims; i++ )
            idx[i] = Range::all();

        idx[axis] = Range(0, 0);

        for( i = 0; i < ninputs; i++ )
        {
            Mat src = inputs.getMat(i);
            idx[axis].end = idx[axis].start + src.size.p[axis];
            Mat buf(dst, idx);
            uchar* data = buf.data;

            src.copyTo(buf);
            CV_Assert(data == buf.data);

            idx[axis].start = idx[axis].end;
        }
    }
};

Ptr<ConcatLayer> ConcatLayer::create(Net& net, const String& name0,
                                     const vector<LayerPin>& inputs, int axis)
{
    String name = net->suggestLayerName(name0, "concat");
    Ptr<ConcatLayer> layer = makePtr<ConcatLayerImpl>(name, (int)inputs.size(), axis);
    net->addLayer(layer, inputs);
    return layer;
}

Ptr<ConcatLayer> ConcatLayer::create(Net& net, const String& name,
                                     const LayerPin& input0, const LayerPin& input1, int axis)
{
    vector<LayerPin> inputs;
    inputs.push_back(input0);
    inputs.push_back(input1);
    return create(net, name, inputs, axis);
}

}
}
