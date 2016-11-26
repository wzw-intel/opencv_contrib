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

#ifndef __OPENCV_DNN_DNN2_HPP__
#define __OPENCV_DNN_DNN2_HPP__

#include <vector>
#include <opencv2/core.hpp>

namespace cv
{
namespace dnn2 //! This namespace is used for dnn module functionlaity.
{
//! @addtogroup dnn
//! @{

    class CV_EXPORTS BaseLayer;
    class CV_EXPORTS BaseNet;
    class CV_EXPORTS LayerPin;

    typedef Ptr<BaseLayer> Layer;
    typedef Ptr<BaseNet> Net;
    using std::vector;

    /** @brief This class allows to create and manipulate comprehensive artificial neural networks.
     *
     * Neural network is presented as directed acyclic graph (DAG), where vertices are Layer instances,
     * and edges specify relationships between layers inputs and outputs.
     *
     * Each network layer has unique integer id and unique string name inside its network.
     * LayerId can store either layer name or layer id.
     *
     * This class supports reference counting of its instances, i. e. copies point to the same instance.
     */
    class CV_EXPORTS_W BaseNet
    {
    public:
        virtual ~BaseNet();

        /** Returns true if there are no layers in the network. */
        CV_WRAP virtual bool empty() const = 0;

        /** Resets the network "finalized" flag and the temporary buffers.
            On the next run the "forward*" method the network topology will be checked
            and the memory buffers will be reallocated.
            Usually, it's not necessary to call this method directly
         */
        CV_WRAP virtual void reset() = 0;

        /** @brief Adds new layer to the net.
         *  @param name   unique name of the adding layer.
         *  @param type   typename of the adding layer (type must be registered in LayerRegister).
         *  @param params parameters which will be used to initialize the creating layer.
         *  @returns unique identifier of created layer, or -1 if a failure will happen.
         *
         *  @details Normally, there is no need to call this function directly.
         *  Layer constructor functions call it automatically
         */
        CV_WRAP virtual void addLayer(const Layer& layer, const vector<LayerPin>& inputs) = 0;
        CV_WRAP virtual void addLayer(const Layer& layer, const LayerPin& input) = 0;

        /** @brief Gets the layer by name
         *  @param name   name of the layer.
         */
        CV_WRAP virtual Layer getLayer(const String& name) const = 0;

        /** @brief Gets all the layers sorted topologically
         *  @param layers the retrieved layers
         */
        CV_WRAP virtual void getLayers(vector<Layer>& layers) const = 0;

        /** @brief Gets all the layers' names. The corresponding layers are sorted topologically
         *  @param names the retrieved names
         */
        CV_WRAP virtual void getLayerNames(vector<String>& names) const = 0;

        /** @brief Gets all the layers' names. The corresponding layers are sorted topologically
         *  @param names the retrieved names
         */
        CV_WRAP virtual Size getImageSize() const = 0;

        /** @brief Generates unique layer name (i.e. there are no layers with such name in the net yet)
         *  @param names the retrieved names
         */
        CV_WRAP virtual String suggestLayerName(const String& name, const String& prefix) const = 0;

        /** @brief Runs the network (forward pass).
         *  @param inputs Input arrays
         */
        CV_WRAP virtual void forward(InputArrayOfArrays inputs) = 0;

        /** @brief Runs the network (forward pass) with a single input.
         *  @param input Input array
         */
        CV_WRAP virtual void forward1(InputArray input) = 0;

        /** @brief Retrieves the particular output
         *  @param outputIdx output index Input array
         */
        CV_WRAP virtual Mat getOutputMat(int idx=0) const = 0;

        /** @brief Retrieves the particular output
         *  @param outputIdx output index
         */
        CV_WRAP virtual int getNumOutputs() const = 0;

        /** @brief Saves the network.
         *  @param arch Name of the file describing network architecture
         *  @param weights Name of the file containing network weights
         */
        CV_WRAP virtual void save(const String& arch, const String& weights) const = 0;

        /** @brief Loads the network.
         *  @param arch Name of the file describing network architecture
         *  @param weights Name of the file containing network weights
         */
        CV_WRAP static Net load(const String& arch, const String& weights);

        /** @brief Creates empty network.
         */
        CV_WRAP static Net create();
    };

    enum
    {
        LAYER_GENERIC = 0,
        LAYER_CONV = 1,
        LAYER_DECONV = 2,
        LAYER_FULL = 3,
        LAYER_ACTIV = 4,
        LAYER_POOLING = 5,
        LAYER_ELEMWISE = 8,
        LAYER_REDUCE = 9,
        LAYER_NORMALIZE = 10,
        LAYER_INPUT = 1024,
        LAYER_OUTPUT = 2048
    };

    /** @brief This interface class allows to build new Layers - are building blocks of networks.
     *
     * Each class, derived from Layer, must implement allocate() methods to declare own outputs and forward() to compute outputs.
     * Also before using the new layer into networks you must register your layer by using one of @ref dnnLayerFactory "LayerFactory" macros.
     */
    class CV_EXPORTS_W BaseLayer
    {
    public:
        virtual ~BaseLayer();

        /** @brief Resets the internal buffers and the finalized flag.
         */
        CV_WRAP virtual void reset() = 0;

        /** @brief Returns true if the layer is finalized.
         */
        CV_WRAP virtual bool isFinalized() const = 0;

        /** @brief Returns the layer's name.
         */
        CV_WRAP virtual String name() const = 0;

        /** @brief Returns the layer's type.
         */
        CV_WRAP virtual int type() const = 0;

        /** @brief Returns the size of the specified output pin
         *  @param inputSizes vector of input array shapes
         *  @param inplaceMask vector of inplace flags
         *       (i-th element is true if i-th input can be processed in-place)
         *  @param outputSizes vector of output array shapes
         *  @param outIdx vector of output array indices:
         *        <0 if i-th output array should be allocated
         *      k>=0 if i-th output array can be put into k-th input array.
         */
        CV_WRAP virtual void finalize(const BaseNet* net,
                                      const vector<vector<int> >& inputSizes,
                                      const vector<bool>& inplaceMask,
                                      vector<vector<int> >& outputSizes,
                                      vector<int>& outIdx,
                                      size_t& bufsize) = 0;

        /** @brief Retrieves the particular output
         *  @param outputIdx output index Input array
         */
        CV_WRAP virtual int getNumOutputs() const = 0;

        /** @brief Processes inputs
         *  @param inputs vector of input arrays
         *  @param outputs vector of output arrays
         *  @param buf temporary buffer, if needed
         */
        CV_WRAP virtual void forward(const BaseNet* net,
                                     InputArrayOfArrays inputs,
                                     OutputArrayOfArrays outputs,
                                     InputOutputArray buf) = 0;

        /* TBD
        virtual void read();
        virtual void write();
        */
    };

    class CV_EXPORTS_W LayerPin
    {
    public:
        LayerPin(const Layer& layer);
        LayerPin(const Layer& layer, int outIdx);

        Layer layer;
        int outIdx;
    };

    CV_EXPORTS_W Net readNetFromCaffe2(const String &prototxt, const String &caffeModel = String());

//! @}
}
}

#include <opencv2/dnn/layers2.hpp>

#endif  /* __OPENCV_DNN_DNN_HPP__ */
