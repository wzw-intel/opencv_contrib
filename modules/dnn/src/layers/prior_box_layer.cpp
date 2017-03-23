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
#include <float.h>
#include <algorithm>
#include <cmath>

namespace cv
{
namespace dnn
{

class PriorBoxLayerImpl : public PriorBoxLayer
{
public:    
    size_t _layerWidth;
    size_t _layerHeight;

    size_t _imageWidth;
    size_t _imageHeight;

    size_t _outChannelSize;

    float _stepX;
    float _stepY;

    float _minSize;
    float _maxSize;

    float _boxWidth;
    float _boxHeight;

    std::vector<float> _aspectRatios;
    std::vector<float> _variance;

    bool _flip;
    bool _clip;

    size_t _numPriors;
    enum { _numAxes = 4 };

    PriorBoxLayerImpl(float minSize, float maxSize, bool flip, bool clip,
                      const std::vector<float>& aspectRatios,
                      const std::vector<float>& variances)
    {
        _minSize = minSize;
        CV_Assert(_minSize > 0);

        _maxSize = maxSize;
        _flip = flip;
        _clip = clip;

        _aspectRatios.clear();
        _aspectRatios.push_back(1.f);

        for( size_t i = 0; i < aspectRatios.size(); i++ )
        {
            float aspectRatio = aspectRatios[i];
            CV_Assert(aspectRatio > 0);

            bool alreadyExists = false;

            for( size_t j = 0; j < _aspectRatios.size(); j++ )
            {
                if (fabs(aspectRatio - _aspectRatios[j]) < 1e-6)
                {
                    alreadyExists = true;
                    break;
                }
            }
            if (!alreadyExists)
            {
                _aspectRatios.push_back(aspectRatio);
                if (_flip)
                    _aspectRatios.push_back(1./aspectRatio);
            }
        }

        _numPriors = _aspectRatios.size();

        if( _maxSize > 0 )
        {
            CV_Assert(_maxSize > _minSize);
            _numPriors += 1;
        }

        size_t nvars = variances.size();
        for( size_t i = 0; i < nvars; i++ )
        {
            float variance = variances[i];
            CV_Assert(variance > 0);
        }
        _variance = variances;
        
        if( !_variance.empty() )
        {
            CV_Assert(nvars == 4 || nvars == 1);
        }
        else
            _variance.push_back(0.1f);
    }

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        CV_Assert(inputs.total() == 2);
        Mat inp0 = inputs.getMat(0), inp1 = inputs.getMat(1);

        _layerWidth = inp0.cols;
        _layerHeight = inp0.rows;

        _imageWidth = inp1.cols;
        _imageHeight = inp1.rows;

        _stepX = static_cast<float>(_imageWidth) / _layerWidth;
        _stepY = static_cast<float>(_imageHeight) / _layerHeight;

        _outChannelSize = _layerHeight * _layerWidth * _numPriors * 4;

        outputs.resizeVector(1);
        // 2 channels. First channel stores the mean of each prior coordinate.
        // Second channel stores the variance of each prior coordinate.
        int outsz[] = { 2, _outChannelSize };
        outputs.create(2, outsz, CV_32F, 0);
    }

    void forward(InputArrayOfArrays, OutputArrayOfArrays outputs)
    {
        Mat outp = outputs.getMat(0);
        float* outputPtr = outp.ptr<float>();

        // first prior: aspect_ratio = 1, size = min_size
        int idx = 0;
        for (size_t h = 0; h < _layerHeight; ++h)
        {
            for (size_t w = 0; w < _layerWidth; ++w)
            {
                _boxWidth = _boxHeight = _minSize;

                float center_x = (w + 0.5) * _stepX;
                float center_y = (h + 0.5) * _stepY;
                // xmin
                outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
                // ymin
                outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
                // xmax
                outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
                // ymax
                outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;

                if (_maxSize > 0)
                {
                    // second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
                    _boxWidth = _boxHeight = sqrt(_minSize * _maxSize);
                    // xmin
                    outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
                    // ymin
                    outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
                    // xmax
                    outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
                    // ymax
                    outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;
                }

                // rest of priors
                for (size_t r = 0; r < _aspectRatios.size(); ++r)
                {
                    float ar = _aspectRatios[r];
                    if (fabs(ar - 1.) < 1e-6)
                    {
                        continue;
                    }
                    _boxWidth = _minSize * sqrt(ar);
                    _boxHeight = _minSize / sqrt(ar);
                    // xmin
                    outputPtr[idx++] = (center_x - _boxWidth / 2.) / _imageWidth;
                    // ymin
                    outputPtr[idx++] = (center_y - _boxHeight / 2.) / _imageHeight;
                    // xmax
                    outputPtr[idx++] = (center_x + _boxWidth / 2.) / _imageWidth;
                    // ymax
                    outputPtr[idx++] = (center_y + _boxHeight / 2.) / _imageHeight;
                }
            }
        }
        // clip the prior's coordidate such that it is within [0, 1]
        if (_clip)
        {
            for (size_t d = 0; d < _outChannelSize; ++d)
            {
                outputPtr[d] = std::min<float>(std::max<float>(outputPtr[d], 0.), 1.);
            }
        }
        // set the variance.
        outputPtr = outp.ptr<float>() + outp.cols;
        if(_variance.size() == 1)
        {
            Mat secondChannel(1, outp.cols, CV_32F, outputPtr);
            secondChannel.setTo(Scalar(_variance[0]));
        }
        else
        {
            int count = 0;
            for (size_t h = 0; h < _layerHeight; ++h)
            {
                for (size_t w = 0; w < _layerWidth; ++w)
                {
                    for (size_t i = 0; i < _numPriors; ++i)
                    {
                        for (int j = 0; j < 4; ++j)
                        {
                            outputPtr[count] = _variance[j];
                            ++count;
                        }
                    }
                }
            }
        }
    }
};

static void getFloatVector(const LayerParams &params, const String& name, std::vector<float>& vec)
{
    DictValue param = params.get(name);
    vec.clear();

    for (int i = 0; i < param.size(); ++i)
    {
        float v = param.get<float>(i);
        vec.push_back(v);
    }
}

Ptr<PriorBoxLayer> PriorBoxLayer::create(const LayerParams &params)
{
    float minSize = params.getParameter<float>("min_size");
    float maxSize = -1.f;
    if (params.has("max_size"))
        maxSize = params.get<float>("max_size");

    float flip = params.getParameter<bool>("flip");
    float clip = params.getParameter<bool>("clip");

    std::vector<float> aspectRatios, variances;
    getFloatVector(params, "aspect_ratio", aspectRatios);
    getFloatVector(params, "variance", variances);

    Ptr<PriorBoxLayer> l(new PriorBoxLayerImpl(minSize, maxSize, flip, clip, aspectRatios, variances));
    l->setParamsFrom(params);

    return l;
}


Ptr<PriorBoxLayer> PriorBoxLayer::create(float minSize, float maxSize, bool flip, bool clip,
                                         const std::vector<float>& aspectRatios,
                                         const std::vector<float>& variances)
{
    return Ptr<PriorBoxLayer>(new PriorBoxLayerImpl(minSize, maxSize, flip, clip, aspectRatios, variances));
}

}
}
