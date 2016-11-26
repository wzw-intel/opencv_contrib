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

class CropLayerImpl : public CropLayer
{
public:
    enum { LTYPE = CV_32F };

    CropLayerImpl(const String& _name, const vector<Vec2i>& _margins)
        : name_(_name), margins(_margins)
    {
        finalized=false;
    }
    virtual ~CropLayerImpl() {}

    String name_;
    vector<Vec2i> margins;
    bool finalized;

    bool isFinalized() const { return finalized; }

    void getMargins(vector<Vec2i>& _margins) const
    {
        _margins = margins;
    }

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
        CV_Assert(inputSizes.size() == 1);
        outputSizes = inputSizes;

        int i, dims = (int)inputSizes[0].size();
        for( i = 0; i < dims; i++ )
        {
            CV_Assert( margins[i][0] >= 0 && margins[i][1] >= 0 &&
                       margins[i][0] + margins[i][1] < inputSizes[0][i] );
            outputSizes[0][i] -= margins[i][0] + margins[i][1];
        }
        outIdx.resize(1, -1);
        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        CV_Assert(src.type() == LTYPE && dst.type() == src.type());

        int i, dims = src.dims;
        AutoBuffer<Range> _idx(dims);
        Range* idx = _idx;

        for( i = 0; i < dims; i++ )
        {
            CV_Assert( dst.size.p[i] == src.size.p[i] + margins[i][0] + margins[i][1] );
            idx[i].start = margins[i][0];
            idx[i].end = src.size.p[i] - margins[i][1];
        }
        Mat buf(src, idx);
        buf.copyTo(dst);
    }
};

Ptr<CropLayer> CropLayer::create(Net& net, const String& name0,
                                 const LayerPin& input, const vector<Vec2i>& margins)
{
    String name = net->suggestLayerName(name0, "crop");
    Ptr<CropLayer> layer = makePtr<CropLayerImpl>(name, margins);
    net->addLayer(layer, input);
    return layer;
}

}
}
