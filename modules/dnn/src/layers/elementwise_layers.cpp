#include "../precomp.hpp"
#include "opencv2/imgproc.hpp"

namespace cv
{
namespace dnn
{

class ReLUFunctor
{
public:
    typedef ReLULayer Layer;
    float slope;

    ReLUFunctor(float slope_) : slope(slope_) {}

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        float s = slope;
        for( size_t i = 0; i < len; i++ )
        {
            float x = src[i];
            dst[i] = x >= 0.f ? x : s*x;
        }
    }
};

class ChannelsPReLUFunctor
{
public:
    typedef ReLULayer Layer;
    Mat slopes;
    const float* slopesptr;

    ChannelsPReLUFunctor(const Mat& slopes_) : slopes(slopes_)
    {
        slopesptr = slopes.ptr<float>();
    }

    bool channelSpecific() const { return true; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int channelIdx) const
    {
        float s = slopesptr[channelIdx];
        for( size_t i = 0; i < len; i++ )
        {
            float x = src[i];
            dst[i] = x >= 0.f ? x : s*x;
        }
    }
};

class TanhFunctor
{
public:
    typedef TanhLayer Layer;

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        for( size_t i = 0; i < len; i++ )
            dst[i] = std::tanh(src[i]);
    }
};

class SigmoidFunctor
{
public:
    typedef SigmoidLayer Layer;

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        for( size_t i = 0; i < len; i++ )
            dst[i] = 1.f/(1.f + std::exp(-src[i]));
    }
};

class AbsFunctor
{
public:
    typedef AbsLayer Layer;

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        for( size_t i = 0; i < len; i++ )
            dst[i] = std::abs(src[i]);
    }
};

class BNLLFunctor
{
public:
    typedef BNLLLayer Layer;

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        for( size_t i = 0; i < len; i++ )
            dst[i] = std::log(1.f + std::exp(-std::abs(src[i])));
    }
};

class PowerFunctor
{
public:
    typedef PowerLayer Layer;
    float power, scale, shift;

    PowerFunctor(float power_, float scale_ = 1.f, float shift_ = 0.f)
    : power(power_), scale(scale_), shift(shift_) {}

    bool channelSpecific() const { return false; }
    int getNInputs() const { return 1; }

    void computePart(const float* src, float* dst, size_t len, int) const
    {
        float p = power, s = scale, sh = shift;
        if( p == 1.f )
        {
            for( size_t i = 0; i < len; i++ )
                dst[i] = s*src[i] + sh;
        }
        else
        {
            for( size_t i = 0; i < len; i++ )
                dst[i] = std::pow(s*src[i] + sh, p);
        }
    }
};

class SumFunctor
{
public:
    typedef SumLayer Layer;
    std::vector<int> coeffs;
    enum { MODE_GENERIC=0, MODE_ADD2=1, MODE_SUB2=2, MODE_RSUB2=3 };
    int ninputs0;
    int mode;

    SumFunctor(const std::vector<int>& coeffs_) : coeffs(coeffs_)
    {
        ninputs0 = (int)coeffs.size();
        mode = MODE_GENERIC;
        if(coeffs.empty() || (ninputs0 >= 2 && coeffs[0] == 1 && coeffs[1] == 1))
        {
            ninputs0 = std::max(2, ninputs0);
            mode = MODE_ADD2;
        }
        else if( ninputs0 >= 2 && (coeffs[0] == 1 && coeffs[1] == -1))
            mode = MODE_SUB2;
        else if( ninputs0 >= 2 && (coeffs[0] == -1 && coeffs[1] == 1))
            mode = MODE_RSUB2;
        else
        {
            CV_Assert(ninputs0 >= 2);
        }
    }

    int getNInputs() const { return ninputs0; }

    void computePart(const float** srcs, int ninputs, float* dst, size_t len) const
    {
        int n = ninputs;
        CV_Assert(ninputs0 == ninputs);
        const float* src0 = srcs[0], *src1 = srcs[1];
        if( mode == MODE_ADD2 )
            for( size_t i = 0; i < len; i++ )
                dst[i] = src0[i] + src1[i];
        else if( mode == MODE_SUB2 )
            for( size_t i = 0; i < len; i++ )
                dst[i] = src0[i] - src1[i];
        else if( mode == MODE_RSUB2 )
            for( size_t i = 0; i < len; i++ )
                dst[i] = src1[i] - src0[i];
        else
        {
            float c0 = (float)coeffs[0], c1 = (float)coeffs[1];
            for( size_t i = 0; i < len; i++ )
                dst[i] = src1[i]*c0 + src0[i]*c1;
        }
        for( int k = 2; k < n; k++ )
        {
            const float* srck = srcs[k];
            int ck = coeffs[k];
            if( ck == 1 )
            {
                for( size_t i = 0; i < len; i++ )
                    dst[i] += srck[i];
            }
            else if( ck == -1 )
            {
                for( size_t i = 0; i < len; i++ )
                    dst[i] -= srck[i];
            }
            else
            {
                for( size_t i = 0; i < len; i++ )
                    dst[i] += (float)ck*srck[i];
            }
        }

    }
};

class ProdFunctor
{
public:
    typedef ProdLayer Layer;

    ProdFunctor()
    {
    }

    int getNInputs() const { return -1; }

    void computePart(const float** srcs, int ninputs, float* dst, size_t len) const
    {
        int n = ninputs;
        const float* src0 = srcs[0], *src1 = srcs[1];
        for( size_t i = 0; i < len; i++ )
            dst[i] = src0[i] * src1[i];
        for( int k = 2; k < n; k++ )
        {
            const float* srck = srcs[k];
            for( size_t i = 0; i < len; i++ )
                dst[i] *= srck[i];
        }
    }
};

class MaxFunctor
{
public:
    typedef MaxLayer Layer;

    MaxFunctor()
    {
    }

    int getNInputs() const { return -1; }

    void computePart(const float** srcs, int ninputs, float* dst, size_t len) const
    {
        int n = ninputs;
        const float* src0 = srcs[0], *src1 = srcs[1];
        for( size_t i = 0; i < len; i++ )
            dst[i] = std::max(src0[i], src1[i]);
        for( int k = 2; k < n; k++ )
        {
            const float* srck = srcs[k];
            for( size_t i = 0; i < len; i++ )
                dst[i] = std::max(dst[i], srck[i]);
        }
    }
};

ElemwiseLayer::~ElemwiseLayer() {}
ElemwiseNAryLayer::~ElemwiseNAryLayer() {}

template<typename Func> class ElemwiseLayerImpl : public Func::Layer
{
public:
    Func func;

    ElemwiseLayerImpl(const Func& func_) : func(func_) {}

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t ninputs = inputs.total();
        CV_Assert(ninputs == (size_t)1);
        outputs.resizeVector(1);
        Mat inp0 = inputs.getMat(0);
        outputs.create(inp0.dims, inp0.size.p, inp0.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t ninputs = inputs.total();
        CV_Assert(ninputs == (size_t)1);

        Mat inp = inputs.getMat(0);
        Mat outp = outputs.getMat(0);

        CV_Assert( inp.type() == CV_32F && inp.isContinuous() );
        CV_Assert( outp.type() == CV_32F && outp.isContinuous() );

        const float* src = inp.ptr<float>();
        float* dst = outp.ptr<float>();
        size_t total = inp.total();
        int j, nblocks = func.channelSpecific() ? inp.size[0] : 1;
        size_t blockSize = total/nblocks;

        for( j = 0; j < nblocks; j++ )
        {
            func.computePart(src, dst, blockSize, j);
            src += blockSize;
            dst += blockSize;
        }
    }

    void computePart(const float* src, float* dst, size_t len, int channelIdx) const
    {
        func.computePart(src, dst, len, channelIdx);
    }
};

template<typename Func> class ElemwiseNAryLayerImpl : public Func::Layer
{
public:
    Func func;

    ElemwiseNAryLayerImpl(const Func& func_) : func(func_) {}

    void allocate(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, ninputs = inputs.total();
        CV_Assert(ninputs == (size_t)func.getNInputs());
        outputs.resizeVector(1);
        Mat inp0 = inputs.getMat(0);
        CV_Assert(inp0.type() == CV_32F);
        for( i = 1; i < ninputs; i++ )
        {
            Mat inp1 = inputs.getMat(i);
            CV_Assert( inp0.size == inp1.size && inp0.type() == inp1.type() );
        }
        outputs.create(inp0.dims, inp0.size.p, inp0.type(), 0);
    }

    void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs)
    {
        size_t i, ninputs = inputs.total();
        CV_Assert(ninputs == (size_t)func.getNInputs());

        Mat inp0 = inputs.getMat(0);
        Mat outp = outputs.getMat(0);

        AutoBuffer<float*> ptrbuf(ninputs+1);
        float** ptrs = ptrbuf;
        size_t total = inp0.total();

        CV_Assert( outp.type() == CV_32F && outp.isContinuous() );

        for( i = 0; i < ninputs; i++ )
        {
            Mat inp = inputs.getMat(i);
            CV_Assert(inp.type() == CV_32F && inp.size == outp.size && inp.isContinuous());
            ptrs[i] = inp.ptr<float>();
        }

        ptrs[ninputs] = outp.ptr<float>();
        func.computePart((const float**)ptrs, ninputs, ptrs[ninputs], total);
    }
    
    void computePart(const float** srcs, int ninputs, float* dst, size_t len) const
    {
        func.computePart(srcs, ninputs, dst, len);
    }
};


Ptr<ReLULayer> ReLULayer::create(float negativeSlope)
{
    return Ptr<ReLULayer>(new ElemwiseLayerImpl<ReLUFunctor>(ReLUFunctor(negativeSlope)));
}

Ptr<ReLULayer> ReLULayer::create(const Mat& slopes)
{
    return Ptr<ReLULayer>(new ElemwiseLayerImpl<ChannelsPReLUFunctor>(ChannelsPReLUFunctor(slopes)));
}

Ptr<ReLULayer> ReLULayer::create(const LayerParams& params)
{
    Ptr<ReLULayer> l;
    if(params.blobs.empty())
    {
        float negativeSlope = params.get<float>("negative_slope", 0.f);
        l = ReLULayer::create(negativeSlope);
    }
    else
    {
        Mat slopes = params.blobs[0];
        l = Ptr<ReLULayer>(new ElemwiseLayerImpl<ChannelsPReLUFunctor>(ChannelsPReLUFunctor(slopes)));
    }
    l->setParamsFrom(params);

    return l;
}

Ptr<TanhLayer> TanhLayer::create()
{
    return Ptr<TanhLayer>(new ElemwiseLayerImpl<TanhFunctor>(TanhFunctor()));
}

Ptr<TanhLayer> TanhLayer::create(const LayerParams& params)
{
    Ptr<TanhLayer> l = TanhLayer::create();
    l->setParamsFrom(params);

    return l;
}

Ptr<SigmoidLayer> SigmoidLayer::create()
{
    return Ptr<SigmoidLayer>(new ElemwiseLayerImpl<SigmoidFunctor>(SigmoidFunctor()));
}

Ptr<SigmoidLayer> SigmoidLayer::create(const LayerParams& params)
{
    Ptr<SigmoidLayer> l = SigmoidLayer::create();
    l->setParamsFrom(params);
    
    return l;
}

Ptr<BNLLLayer> BNLLLayer::create()
{
    return Ptr<BNLLLayer>(new ElemwiseLayerImpl<BNLLFunctor>(BNLLFunctor()));
}

Ptr<BNLLLayer> BNLLLayer::create(const LayerParams& params)
{
    Ptr<BNLLLayer> l = BNLLLayer::create();
    l->setParamsFrom(params);
    
    return l;
}

Ptr<AbsLayer> AbsLayer::create()
{
    return Ptr<AbsLayer>(new ElemwiseLayerImpl<AbsFunctor>(AbsFunctor()));
}

Ptr<AbsLayer> AbsLayer::create(const LayerParams& params)
{
    Ptr<AbsLayer> l = AbsLayer::create();
    l->setParamsFrom(params);
    
    return l;
}

Ptr<PowerLayer> PowerLayer::create(float power, float scale, float shift)
{
    return Ptr<PowerLayer>(new ElemwiseLayerImpl<PowerFunctor>(PowerFunctor(power, scale, shift)));
}

Ptr<PowerLayer> PowerLayer::create(const LayerParams& params)
{
    float power = params.get<float>("power", 1.0f);
    float scale = params.get<float>("scale", 1.0f);
    float shift = params.get<float>("shift", 0.0f);
    Ptr<PowerLayer> l = PowerLayer::create(power, scale, shift);
    l->setParamsFrom(params);

    return l;
}

Ptr<SumLayer> SumLayer::create(const std::vector<int> &coeffs)
{
    return Ptr<SumLayer>(new ElemwiseNAryLayerImpl<SumFunctor>(SumFunctor(coeffs)));
}

Ptr<ProdLayer> ProdLayer::create()
{
    return Ptr<ProdLayer>(new ElemwiseNAryLayerImpl<ProdFunctor>(ProdFunctor()));
}

Ptr<MaxLayer> MaxLayer::create()
{
    return Ptr<MaxLayer>(new ElemwiseNAryLayerImpl<MaxFunctor>(MaxFunctor()));
}


Ptr<ElemwiseNAryLayer> ElemwiseNAryLayer::create(const LayerParams& params)
{
    String operation = params.has("operation") ?
    params.get<String>("operation").toLowerCase() : "sum";

    std::vector<int> coeffs;
    if (params.has("coeff"))
    {
        DictValue paramCoeff = params.get("coeff");
        int i, n = paramCoeff.size();
        coeffs.resize(n);
        for (i = 0; i < n; i++)
            coeffs[i] = paramCoeff.get<int>(i);
    }

    Ptr<ElemwiseNAryLayer> layer;
    if (operation == "prod")
    {
        CV_Assert(coeffs.empty());
        layer = ProdLayer::create();
    }
    else if (operation == "sum")
        layer = SumLayer::create(coeffs);
    else if (operation == "max")
    {
        CV_Assert(coeffs.empty());
        layer = MaxLayer::create();
    }
    else
        CV_Error(cv::Error::StsBadArg, "Unknown operaticon type \"" + operation + "\"");
    
    return layer;
}

}
}
