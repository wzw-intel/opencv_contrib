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
#include "op_blas.hpp"

namespace cv
{
namespace dnn
{

class NormalizeBBoxLayerImpl : public NormalizeBBoxLayer
{
public:
    Mat _buffer;

    Mat _sumChannelMultiplier;
    Mat _sumSpatialMultiplier;

    Mat _scale;

    float _eps;
    bool _across_spatial;
    bool _channel_shared;

    size_t _num;
    size_t _channels;
    size_t _rows;
    size_t _cols;

    size_t _channelSize;
    size_t _imageSize;

    static const size_t _numAxes = 4;
    static const std::string _layerName;

    NormalizeBBoxLayerImpl(bool acrossSpatial, bool channelShared, float eps)
    {
        _eps = eps;
        _across_spatial = acrossSpatial;
        _channel_shared = channelShared;
    }

    void checkInputs(InputArrayOfArrays inputs)
    {
        size_t i, ninputs = inputs.total();
        CV_Assert(ninputs > 0);
        Mat inp0 = inputs.getMat(0);
        CV_Assert(inp0.dims > 2);

        for (i = 1; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            CV_Assert(inp.size == inp0.size);
        }
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        checkInputs(inputs);
        Mat inp0 = inputs.getMat(0);

        _num = 1;
        _channels = inp0.size[0];
        _rows = inp0.size[1];
        _cols = inp0.size[2];

        _channelSize = _rows * _cols;
        _imageSize = _channelSize * _channels;

        _buffer = Mat(_channels, _channelSize, CV_32F);

        _sumChannelMultiplier = Mat(_channels, 1, CV_32F, Scalar(1.0));
        _sumSpatialMultiplier = Mat(1, _channelSize, CV_32F, Scalar(1.0));

        _scale = blobs[0];

        size_t i, ninputs = inputs.total();
        std::vector<Mat>& outp = outputs.getMatVecRef();
        outp.resize(ninputs);

        for(i = 0; i < ninputs; i++)
        {
            Mat inp = inputs.getMat(i);
            outp[i].create(inp.dims, inp.size.p, inp.type());
        }
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t j, ninputs = inputs.total();
        Mat zeroBuffer(_channels, _channelSize, CV_32F, Scalar(0));
        Mat absDiff;

        for (j = 0; j < ninputs; j++)
        {
            Mat inp = inputs.getMat(j);
            Mat outp = outputs.getMat(j);
            Mat src = Mat(_channels, _channelSize, CV_32F, inp.ptr<float>());
            Mat dst = Mat(_channels, _channelSize, CV_32F, outp.ptr<float>());

            _buffer = src.mul(src);

            if (_across_spatial)
            {
                absdiff(_buffer, zeroBuffer, absDiff);

                // add eps to avoid overflow
                double absSum = sum(absDiff)[0] + _eps;

                float norm = sqrt(absSum);
                dst = src / norm;
            }
            else
            {
                Mat norm(_channelSize, 1, _buffer.type()); // 1 x _channelSize

                // (_channels x_channelSize)T * _channels x 1 -> _channelSize x 1
                gemmCPU(_buffer, _sumChannelMultiplier, 1, norm, 0, GEMM_1_T);

                // compute norm
                pow(norm, 0.5f, norm);

                // scale the layer
                // _channels x 1 * (_channelSize x 1)T -> _channels x _channelSize
                gemmCPU(_sumChannelMultiplier, norm, 1, _buffer, 0, GEMM_2_T);
                
                dst = src / _buffer;
            }
            
            // scale the output
            if (_channel_shared)
            {
                // _scale: 1 x 1
                dst *= _scale.at<float>(0, 0);
            }
            else
            {
                // _scale: _channels x 1
                // _channels x 1 * 1 x _channelSize -> _channels x _channelSize
                gemmCPU(_scale, _sumSpatialMultiplier, 1, _buffer, 0);
                
                dst = dst.mul(_buffer);
            }
        }
    }
};

Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(bool acrossSpatial, bool channelShared, float eps)
{
    return Ptr<NormalizeBBoxLayer>(new NormalizeBBoxLayerImpl(acrossSpatial, channelShared, eps));
}

Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(const LayerParams& params)
{
    double eps = params.getParameter<float>("eps", 0, false, 1e-10f);
    bool acrossSpatial = params.getParameter<bool>("across_spatial");
    bool channelShared = params.getParameter<bool>("channel_shared");

    Ptr<NormalizeBBoxLayer> l(new NormalizeBBoxLayerImpl(acrossSpatial, channelShared, (float)eps));
    l->setParamsFrom(params);

    return l;
}

}
}
