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

#include "precomp.hpp"
#include <map>
#include <set>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

namespace cv {
namespace dnn2 {

using std::map;
using std::vector;

class NetImpl : public BaseNet
{
public:
    vector<Layer> layers;
    typedef map<String, int> lmap_t;
    lmap_t layerNames;
    vector<Vec2i> netOutputs;
    vector<vector<int> > inputSizes0;
    vector<vector<int> > nconsumers;
    vector<vector<Vec2i> > topology;
    vector<Mat> matpool;
    vector<vector<vector<int> > > outputSizes;
    vector<Mat> inputs0;
    vector<vector<Mat> > inputs;
    vector<vector<Mat> > outputs;
    Mat tempbuf;
    bool finalized;
    Size imgSize0;
    int channels0;
    int netinputs;

    NetImpl()
    {
        netinputs = 0;
        channels0 = 1;
        finalized = false;
    }
    virtual ~NetImpl() {}
    bool empty() const { return layers.empty(); }

    void addLayer(const Layer& layer, const vector<LayerPin>& inputs)
    {
        finalized = false;
        String name = layer->name();
        if(layerNames.find(name) != layerNames.end())
            CV_Error(Error::StsError, "the layer with the same name already exists in the network");
        int l_idx = (int)layers.size();
        layers.push_back(layer);

        layerNames[name] = l_idx;
        topology.push_back(vector<Vec2i>());

        int noutputs = layer->getNumOutputs();
        vector<int> nconsumers_l(noutputs, 0);
        nconsumers.push_back(nconsumers_l);
        vector<Vec2i>& t = topology.back();

        size_t i, ninputs = inputs.size();
        bool isinput = (layer->type() & LAYER_INPUT) != 0;
        if( isinput )
        {
            netinputs++;
            if( netinputs != (int)layers.size() )
                CV_Error(Error::StsBadFlag,
                         "The input layers must be the first layers added to the network. "
                         "You may not add an input layer after some non-input layer");
        }
        for( i = 0; i < ninputs; i++ )
        {
            const Layer& il = inputs[i].layer;
            String name_i = il->name();
            lmap_t::const_iterator it = layerNames.find(name_i);
            if(it == layerNames.end())
                CV_Error(Error::StsError, "one of layer's inputs is not in the net yet");
            int layerIdx = it->second;
            int outIdx = inputs[i].outIdx;
            CV_Assert( 0 <= layerIdx && layerIdx < (int)nconsumers.size() );
            CV_Assert( 0 <= outIdx && outIdx < il->getNumOutputs());
            nconsumers[layerIdx][outIdx]++;
            t.push_back(Vec2i(layerIdx, outIdx));
        }
    }

    void addLayer(const Layer& layer, const LayerPin& input)
    {
        vector<LayerPin> inputs;
        inputs.push_back(input);
        addLayer(layer, inputs);
    }

    Layer getLayer(const String& name) const
    {
        size_t i, nlayers = layers.size();
        for( i = 0; i < nlayers; i++ )
        {
            if( layers[i]->name() == name )
                return layers[i];
        }
        return Layer();
    }

    void getLayers(vector<Layer>& outlayers) const
    {
        outlayers.clear();
        std::copy(layers.begin(), layers.end(), std::back_inserter(outlayers));
    }

    void getLayerNames(vector<String>& names) const
    {
        size_t i, nlayers = layers.size();
        names.resize(nlayers);

        for( i = 0; i < nlayers; i++ )
            names[i] = layers[i]->name();
    }

    Size getImageSize() const
    {
        return imgSize0;
    }

    String suggestLayerName(const String& name0, const String& prefix) const
    {
        if(name0.size() > 0)
            return name0;
        int val = 1;
        String name;

        for(;; val++)
        {
            name = format("%s%d", prefix.c_str(), val);
            if(layerNames.find(name) != layerNames.end())
                break;
        }
        return name;
    }

    void updateNetInputSizes(const Mat* inputs, size_t ninputs)
    {
        CV_Assert((int)ninputs == netinputs && netinputs > 0);
        inputSizes0.resize(ninputs);
        for( size_t i = 0; i < ninputs; i++ )
        {
            int j, dims = inputs[i].dims;
            const int* sz = &inputs[i].size.p[0];
            inputSizes0[i].resize(dims+1);

            for( j = 0; j < dims; j++ )
                inputSizes0[i][j+1] = sz[j];
            inputSizes0[i][0] = inputs[i].channels();
        }

        if(imgSize0 != Size() )
            CV_Assert(netinputs == 1 &&
                      inputs[0].size() == imgSize0 &&
                      inputs[0].channels() == channels0);
    }

    void reset()
    {
        netOutputs.clear();
        inputSizes0.clear();
        matpool.clear();
        outputSizes.clear();
        inputs.clear();
        outputs.clear();
        tempbuf.release();
        imgSize0 = Size();
    }

    int getMatBuf(size_t sz, vector<size_t>& matsize, vector<int>& matconsumers)
    {
        int j, n = (int)matsize.size();
        int k = -1;
        size_t delta = INT_MAX;
        for( j = 0; j < n; j++ )
        {
            if( matconsumers[j] == 0 && (k == -1 || (matsize[j] >= sz && matsize[j] - sz < delta)) )
            {
                k = j;
                if( matsize[j] >= sz )
                    delta = matsize[j] - sz;
                if( delta == 0 )
                    break;
            }
        }

        if( k < 0 )
        {
            matsize.push_back(sz);
            matconsumers.push_back(0);
            k = n;
        }
        else
            matsize[k] = max(matsize[k], sz);
        return k;
    }

    void finalize(const Mat* netinputs, size_t ninputs0)
    {
        const int mattype = CV_32F;

        // if the network is already finalized, we need to check
        // if the input arrays have the same size as before;
        // if not, we need to reallocate the buffers.
        // some networks, e.g. semantical segmentation,
        // should be able to process variable-size arrays
        bool f = finalized;
        size_t i, nlayers = layers.size();
        for( i = 0; i < nlayers; i++ )
        {
            if(!f)
                break;
            f = layers[i]->isFinalized();
        }

        if( f && ninputs0 == inputSizes0.size() )
        {
            for( i = 0; i < ninputs0; i++ )
            {
                int j = 0, dims = netinputs[i].dims;
                const int* sz = &netinputs[i].size.p[0];
                if( dims+1 != (int)inputSizes0[i].size() )
                    break;
                const int* sz0 = &inputSizes0[i][0];
                if( netinputs[i].channels() != sz0[0] )
                    break;
                for( ; j < dims; j++ )
                    if(sz[j] != sz0[j+1] )
                        break;
                if( j < dims )
                    break;
            }
            if( i == ninputs0 )
                return;
        }

        // We get here if either:
        //   1. a new layer or a few have been added to the network
        //   2. some of the layers have been changed
        //   3. the size of input image(s) have been changed
        // in the case 2 each layer has some freedom in resetting or not
        // the "finalized" flag when some of its parameter change.
        //
        // Some of the parameters can, in fact, be changed on fly
        // without reallocation of re-finalization of the whole network
        // (e.g. the negative slope of the ReLU function).

        // We reset all the internal buffers and assign them again.
        reset();
        updateNetInputSizes(netinputs, ninputs0);

        vector<vector<int> > inputSizes;
        vector<bool> allowInplace;
        vector<vector<int> > outIdx;
        vector<int> matconsumers;
        vector<size_t> matsize;
        vector<int> inputIdx;

        size_t maxBufSize = 0;

        outputSizes.resize(nlayers);
        outIdx.resize(nlayers);

        /*
        The next step is to create the pool of matrixes needed to run the whole network, the "matpool".
        The matpool may be viewed as a set of "registers" in which we store inputs/outputs of each layer.
        The following loop assigns each input/output array of each layer to one of the "registers".
        Each register, i.e. matpool[k] for some k, is a matrix of certain size. During the network
        execution we will store various arrays in it, and those arrays may have different size.
        We do it by dynamically cropping and reshaping matpool[k] using Mat::fit() method.

        The loop uses two precomputed vectors that define the network topology:
           * "topology" vector. topology[i][j] is a pair (k, l).
                    It indicates that j-th input of i-th layer is l-th output of k-th layer.
           * "nconsumers" vector. nconsumers[i][j] == k indicates that the j-th output of i-th layer
                    is used k times downstream in the network, i.e. there are k "consumers" of it.
        Strictly speaking, nconsumers[i][j] could be computed on-fly from "topology", but instead
        we store nconsumers and update it as we add new layers into the network; see addLayer() method.
     
        After the loop we have the following elements computed:
           * the amount of registers needed to run the network. This is "matsize.size()".
           * "outIdx" vector. outIdx[i][j] == k indicates that j-th output of i-th layer
                    will be stored in matpool[k].
           * "outputSizes" vector. outputSizes[i][j] is the shape of j-th output of i-th layer.
           * "matsize" vector. matsize[k] is the size of matpool[k] (in float's)
           * maxBufSize - the maximum size of the temporary buffer.
             Temporary buffer is used by some of the layers like
             convolution, deconvolution, fully-connected layer etc.
     
        These are important elements of the assignment algorithm logic:
           * "matconsumers" vector, maintained during the whole loop.
              matconsumers[k]=n indicates that currently k-th register holds
              an array that will be used "n" times downstream in the network.
              If matconsumers[k]==0, it means that k-th register is "free" and
              could be assigned some output of the currently observed layer.

           * "allowInplace" vector, constructed from scratch on each iteration.
             allowInplace[j] == true if j-th input of currently observed layer is not
             used anywhere further except for the layer, so the layer can overwrite it.
             Obvisouly, as we know that j-th input is kept in some k_j-th "register",
             we compute allowInplace[k_j] as: "allowInplace[k_j] = matconsumers[k_j] == 1".

           * virtual method BaseLayer::finalize() that we call for each layer. It takes
             the vector of sizes of all the layer inputs and also the "allowInplace" vector and
             computes the output array sizes and the decision on each output j - whether it will
             reside in the same buffer as some k-th input (outIdx[i][j] == k) or not (outIdx[i][j] < 0).

               ** In the first case we overwrite outIdx[i][j] with the actual "register" index (stored in inputIdx[k]).
               ** In the second case we call getMatBuf() method that finds the most appropriate free "register" to keep
                  the array.
                  *** If there are free registers, but they all have insufficient size,
                      we expand one of them to fit the layer output.
                  *** If there are no free registers at all, we add a new register
                      (i.e. expand matsize and matconsumers).

           * BaseLayer::finalize() also gives us the buffer size used by i-th layer.
             We update the maximum buffer size so that it fits any temporary buffer (assuming that
             we process the layers strictly sequentially, one by one, not in parallel threads).

           * after we are done with i-th layer (i.e. computed outIdx[i][*]):
              ** we decrement matconsumers[k] if k-th register is passed as input to i-th layer, but is not processed in-place.
              ** we replace matconsumers[k] with nconsumers[i][j] if k-th register holds the j-th output of i-th layer.
        */
        for( i = 0; i < nlayers; i++ )
        {
            size_t j, noutputs = nconsumers[i].size();
            for( j = 0; j < noutputs; j++ )
                if(nconsumers[i][j] == 0)
                    netOutputs.push_back(Vec2i(i, j));
            size_t ninputs = topology[i].size();
            inputSizes.resize(ninputs);
            inputIdx.resize(ninputs);
            allowInplace.resize(ninputs);

            bool isinput = i < ninputs0;
            if(isinput)
            {
                CV_Assert(ninputs == 1);
                Mat img = netinputs[i];
                inputSizes[0].resize(3);
                inputSizes[0][0] = img.channels();
                inputSizes[0][1] = img.rows;
                inputSizes[0][2] = img.cols;
                inputIdx[0] = -1;
                allowInplace[0] = false;
            }
            else
            {
                for( j = 0; j < ninputs; j++ )
                {
                    Vec2i pin = topology[i][j];
                    inputSizes[j] = outputSizes[pin[0]][pin[1]];
                    int oidx = outIdx[pin[0]][pin[1]];
                    inputIdx[j] = oidx;
                    allowInplace[j] = matconsumers[oidx] == 1;
                }
            }

            size_t bufsize = 0;
            outputSizes[i].clear();
            outIdx[i].clear();
            layers[i]->finalize(this, inputSizes, allowInplace, outputSizes[i], outIdx[i], bufsize);
            maxBufSize = max(maxBufSize, bufsize);

            CV_Assert(outIdx[i].size() == noutputs);
            for( j = 0; j < noutputs; j++ )
            {
                int oidx = outIdx[i][j];
                int delta = 0;
                if( oidx >= 0 )
                {
                    CV_Assert( oidx < (int)ninputs );
                    if( !allowInplace[oidx] )
                        oidx = -1;
                }

                if( oidx >= 0 )
                {
                    oidx = inputIdx[oidx];
                    delta = 1;
                }
                else
                {
                    size_t bufsz = matSize(outputSizes[i][j]);
                    oidx = getMatBuf(bufsz, matsize, matconsumers);
                }

                outIdx[i][j] = oidx;
                int counter = nconsumers[i][j];
                matconsumers[oidx] = counter > 0 ? counter+delta : INT_MAX;
            }

            if(!isinput)
                for( j = 0; j < ninputs; j++ )
                {
                    Vec2i pin = topology[i][j];
                    int oidx = outIdx[pin[0]][pin[1]];
                    matconsumers[oidx]--;
                }
        }

        /*
           After we are done with the sophisticated assignment part, we just need to:
           1. allocate "matpool" according to the computed "matsize" vector.
           2. constuct array headers for all the layers' input/outputs:
              basically, outputs[i][j] will reside in outIdx[i][j]-th register
              and will have outputSizes[i][j] size. Then, if topology[i][j] == (k, l),
              then inputs[i][j] = outputs[k][l].
        */
        CV_Assert( maxBufSize <= (size_t)INT_MAX );
        tempbuf.create(1, (int)maxBufSize, mattype);

        inputs.resize(nlayers);
        outputs.resize(nlayers);

        size_t nmats = matsize.size();
        matpool.resize(nmats);

        for( i = 0; i < nmats; i++ )
        {
            CV_Assert( matsize[i] <= (size_t)INT_MAX );
            int sz = (int)matsize[i];
            matpool[i].create(1, sz, mattype);
        }

        for( i = 0; i < nlayers; i++ )
        {
            size_t j, noutputs = nconsumers[i].size();
            size_t ninputs = topology[i].size();
            outputs[i].resize(noutputs);
            inputs[i].resize(ninputs);

            for( j = 0; j < noutputs; j++ )
            {
                Mat& mtx = matpool[outIdx[i][j]];
                outputs[i][j] = mtx;
                int dims = (int)outputSizes[i][j].size();
                const uchar* data = mtx.data;
                outputs[i][j].fit(dims, &outputSizes[i][j][0], mattype);
                CV_Assert(mtx.data == data);
            }

            for( j = 0; j < ninputs; j++ )
            {
                Vec2i pin = topology[i][j];
                inputs[i][j] = outputs[pin[0]][pin[1]];
            }
        }

        finalized = true;
    }

    void forward(const Mat* inputs, size_t ninputs)
    {
        finalize(inputs, ninputs);
        inputs0.resize(1);

        size_t i, nlayers = topology.size();
        // the first "ninputs" layers are input layers;
        // copy the input images to those layers
        for( i = 0; i < ninputs; i++ )
        {
            inputs0[0] = inputs[i];
            layers[i]->forward(this, inputs0, outputs[i], tempbuf);
        }

        // process the network
        for( ; i < nlayers; i++ )
            layers[i]->forward(this, inputs[i], outputs[i], tempbuf);

        inputs0[0] = Mat();
    }

    void forward(InputArrayOfArrays inputs)
    {
        vector<Mat> mv;
        inputs.getMatVector(mv);
        forward(mv.empty() ? 0 : &mv[0], mv.size());
    }

    void forward1(InputArray input)
    {
        Mat input0 = input.getMat();
        forward(&input0, 1);
    }

    Mat getOutputMat(int idx=0) const
    {
        CV_Assert( 0 <= idx && idx < (int)netOutputs.size());
        Vec2i idx2 = netOutputs[idx];
        return outputs[idx2[0]][idx2[1]];
    }

    int getNumOutputs() const
    {
        return (int)netOutputs.size();
    }

    void save(const String& /*arch*/, const String& /*weights*/) const
    {

    }
};

BaseNet::~BaseNet() {}
Net BaseNet::create()
{
    Net net = makePtr<NetImpl>();
    return net;
}

Net BaseNet::load(const String& arch, const String& weights)
{
    Net net = BaseNet::create();
    return net;
}

LayerPin::LayerPin(const Layer& _layer) : layer(_layer), outIdx(0)
{
    CV_Assert(layer->getNumOutputs()==1);
}

LayerPin::LayerPin(const Layer& _layer, int _outIdx) : layer(_layer), outIdx(_outIdx)
{
    CV_Assert(0 <= outIdx && outIdx < layer->getNumOutputs());
}

BaseLayer::~BaseLayer() {}

}
}
