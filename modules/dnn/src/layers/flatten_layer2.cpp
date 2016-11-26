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

class FlattenLayerImpl : public FlattenLayer
{
public:
    enum { LTYPE = CV_32F };

    FlattenLayerImpl(const String& _name, const Range& _axisRange) :
    name_(_name), axisRange(_axisRange)
    {
        finalized=false;
    }
    virtual ~FlattenLayerImpl() {}

    String name_;
    Range axisRange;
    bool finalized;

    bool isFinalized() const { return finalized; }
    Range getAxisRange() const { return axisRange; }

    void reset()
    {
        finalized = false;
    }

    String name() const { return name_; }
    int type() const { return LAYER_GENERIC; }

    int getOutputSize(const int* inputSizes, int idims,
                       int* outputSizes)
    {
        int j, odims = 2;
        Range ar = axisRange;
        if( ar == Range::all() )
            ar = Range(0, idims);

        CV_Assert( ar.start >= 0 && ar.end > ar.start && ar.end <= idims );

        odims = idims - (ar.end - ar.start) + 1;
        for( j = 0; j < ar.start; j++ )
            outputSizes[j] = inputSizes[j];
        int sz = 1;
        for( ; j < ar.end; j++ )
            sz *= inputSizes[j];
        outputSizes[ar.start] = sz;
        for( ; j < idims; j++ )
            outputSizes[j + odims - idims] = inputSizes[j];
        if( odims == 1 )
            outputSizes[odims++] = 1;
        return odims;
    }

    void finalize(const BaseNet*,
                  const vector<vector<int> >& inputSizes,
                  const vector<bool>& inplaceMask,
                  vector<vector<int> >& outputSizes,
                  vector<int>& outIdx,
                  size_t&)
    {
        CV_Assert(inputSizes.size() == 1);
        outputSizes.resize(1);
        outputSizes[0].resize(CV_MAX_DIM);
        int idims = (int)inputSizes[0].size();
        int odims = getOutputSize(&inputSizes[0][0], idims, &outputSizes[0][0]);
        outputSizes[0].resize(odims);
        outIdx.resize(1, inplaceMask[0] ? 0 : -1);

        finalized = true;
    }

    int getNumOutputs() const { return 1; }

    void forward(const BaseNet*, InputArrayOfArrays inputs,
                 OutputArrayOfArrays outputs, InputOutputArray)
    {
        Mat src = inputs.getMat(0);
        Mat dst = outputs.getMat(0);
        int osize[CV_MAX_DIM];
        int idims = src.dims, odims = getOutputSize(src.size.p, idims, osize);

        CV_Assert(src.type() == LTYPE && dst.type() == src.type() &&
                  dst.dims == odims && src.total() == dst.total());
        CV_Assert(src.isContinuous() && dst.isContinuous());

        if( dst.data != src.data )
        {
            Mat dstAlias(idims, src.size.p, LTYPE, dst.data);
            src.copyTo(dstAlias);
        }
    }
};

Ptr<FlattenLayer> FlattenLayer::create(Net& net, const String& name0, const LayerPin& input,
                                       const Range& axisRange)
{
    String name = net->suggestLayerName(name0, "flatten");
    Ptr<FlattenLayer> layer = makePtr<FlattenLayerImpl>(name, axisRange);
    net->addLayer(layer, input);
    return layer;
}

}
}
