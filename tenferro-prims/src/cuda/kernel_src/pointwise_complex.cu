template <typename T>
struct TfComplex {
    T re;
    T im;
};

template <typename T>
struct ScalarOps;

template <>
struct ScalarOps<float> {
    __device__ static float zero() { return 0.0f; }
    __device__ static float one() { return 1.0f; }
    __device__ static float half() { return 0.5f; }
    __device__ static float pi() { return 3.14159265358979323846f; }
    __device__ static float abs(float x) { return ::fabsf(x); }
    __device__ static float atan2(float y, float x) { return ::atan2f(y, x); }
    __device__ static float cos(float x) { return ::cosf(x); }
    __device__ static float cosh(float x) { return ::coshf(x); }
    __device__ static float copy_sign(float x, float y) { return ::copysignf(x, y); }
    __device__ static float exp(float x) { return ::expf(x); }
    __device__ static float hypot(float x, float y) { return ::hypotf(x, y); }
    __device__ static float log(float x) { return ::logf(x); }
    __device__ static float sin(float x) { return ::sinf(x); }
    __device__ static float sinh(float x) { return ::sinhf(x); }
    __device__ static float sqrt(float x) { return ::sqrtf(x); }
};

template <>
struct ScalarOps<double> {
    __device__ static double zero() { return 0.0; }
    __device__ static double one() { return 1.0; }
    __device__ static double half() { return 0.5; }
    __device__ static double pi() { return 3.14159265358979323846; }
    __device__ static double abs(double x) { return ::fabs(x); }
    __device__ static double atan2(double y, double x) { return ::atan2(y, x); }
    __device__ static double cos(double x) { return ::cos(x); }
    __device__ static double cosh(double x) { return ::cosh(x); }
    __device__ static double copy_sign(double x, double y) { return ::copysign(x, y); }
    __device__ static double exp(double x) { return ::exp(x); }
    __device__ static double hypot(double x, double y) { return ::hypot(x, y); }
    __device__ static double log(double x) { return ::log(x); }
    __device__ static double sin(double x) { return ::sin(x); }
    __device__ static double sinh(double x) { return ::sinh(x); }
    __device__ static double sqrt(double x) { return ::sqrt(x); }
};

enum UnaryOp : int {
    UNARY_NEG = 0,
    UNARY_CONJ = 1,
    UNARY_ABS = 2,
    UNARY_RECIPROCAL = 3,
    UNARY_REAL = 4,
    UNARY_IMAG = 5,
    UNARY_SQRT = 6,
    UNARY_RSQRT = 7,
    UNARY_EXP = 8,
    UNARY_EXPM1 = 9,
    UNARY_LOG = 10,
    UNARY_LOG1P = 11,
    UNARY_SIN = 12,
    UNARY_COS = 13,
    UNARY_TAN = 14,
    UNARY_TANH = 15,
    UNARY_ASIN = 16,
    UNARY_ACOS = 17,
    UNARY_ATAN = 18,
    UNARY_SINH = 19,
    UNARY_COSH = 20,
    UNARY_ASINH = 21,
    UNARY_ACOSH = 22,
    UNARY_ATANH = 23
};

enum BinaryOp : int {
    BINARY_SUB = 0,
    BINARY_DIV = 1,
    BINARY_POW = 2,
    BINARY_XLOGY = 3
};

template <typename T>
__device__ inline TfComplex<T> tf_complex(T re, T im) {
    TfComplex<T> out{re, im};
    return out;
}

template <typename T>
__device__ inline TfComplex<T> complex_zero() {
    return tf_complex<T>(ScalarOps<T>::zero(), ScalarOps<T>::zero());
}

template <typename T>
__device__ inline TfComplex<T> complex_one() {
    return tf_complex<T>(ScalarOps<T>::one(), ScalarOps<T>::zero());
}

template <typename T>
__device__ inline TfComplex<T> complex_from_real(T re) {
    return tf_complex<T>(re, ScalarOps<T>::zero());
}

template <typename T>
__device__ inline bool complex_is_zero(TfComplex<T> z) {
    return z.re == ScalarOps<T>::zero() && z.im == ScalarOps<T>::zero();
}

template <typename T>
__device__ inline TfComplex<T> complex_add(TfComplex<T> x, TfComplex<T> y) {
    return tf_complex<T>(x.re + y.re, x.im + y.im);
}

template <typename T>
__device__ inline TfComplex<T> complex_sub(TfComplex<T> x, TfComplex<T> y) {
    return tf_complex<T>(x.re - y.re, x.im - y.im);
}

template <typename T>
__device__ inline TfComplex<T> complex_mul(TfComplex<T> x, TfComplex<T> y) {
    return tf_complex<T>(x.re * y.re - x.im * y.im, x.re * y.im + x.im * y.re);
}

template <typename T>
__device__ inline TfComplex<T> complex_scale(TfComplex<T> x, T scale) {
    return tf_complex<T>(x.re * scale, x.im * scale);
}

template <typename T>
__device__ inline TfComplex<T> complex_conj(TfComplex<T> x) {
    return tf_complex<T>(x.re, -x.im);
}

template <typename T>
__device__ inline T complex_abs_value(TfComplex<T> x) {
    return ScalarOps<T>::hypot(x.re, x.im);
}

template <typename T>
__device__ inline TfComplex<T> complex_div(TfComplex<T> x, TfComplex<T> y) {
    T denom = y.re * y.re + y.im * y.im;
    return tf_complex<T>(
        (x.re * y.re + x.im * y.im) / denom,
        (x.im * y.re - x.re * y.im) / denom
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_inv(TfComplex<T> x) {
    T denom = x.re * x.re + x.im * x.im;
    return tf_complex<T>(x.re / denom, -x.im / denom);
}

template <typename T>
__device__ inline TfComplex<T> complex_exp(TfComplex<T> x) {
    T scale = ScalarOps<T>::exp(x.re);
    return tf_complex<T>(
        scale * ScalarOps<T>::cos(x.im),
        scale * ScalarOps<T>::sin(x.im)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_log(TfComplex<T> x) {
    return tf_complex<T>(
        ScalarOps<T>::log(complex_abs_value(x)),
        ScalarOps<T>::atan2(x.im, x.re)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_sqrt(TfComplex<T> x) {
    if (complex_is_zero(x)) {
        return complex_zero<T>();
    }

    T magnitude = complex_abs_value(x);
    if (x.re >= ScalarOps<T>::zero()) {
        T real = ScalarOps<T>::sqrt((magnitude + x.re) * ScalarOps<T>::half());
        T imag = x.im / (real + real);
        return tf_complex<T>(real, imag);
    }

    T imag_mag = ScalarOps<T>::sqrt((magnitude - x.re) * ScalarOps<T>::half());
    T real = ScalarOps<T>::abs(x.im) / (imag_mag + imag_mag);
    T imag_sign = x.im == ScalarOps<T>::zero() ? ScalarOps<T>::one() : x.im;
    return tf_complex<T>(real, ScalarOps<T>::copy_sign(imag_mag, imag_sign));
}

template <typename T>
__device__ inline TfComplex<T> complex_mul_i(TfComplex<T> x) {
    return tf_complex<T>(-x.im, x.re);
}

template <typename T>
__device__ inline TfComplex<T> complex_mul_neg_i(TfComplex<T> x) {
    return tf_complex<T>(x.im, -x.re);
}

template <typename T>
__device__ inline TfComplex<T> complex_sin(TfComplex<T> x) {
    return tf_complex<T>(
        ScalarOps<T>::sin(x.re) * ScalarOps<T>::cosh(x.im),
        ScalarOps<T>::cos(x.re) * ScalarOps<T>::sinh(x.im)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_cos(TfComplex<T> x) {
    return tf_complex<T>(
        ScalarOps<T>::cos(x.re) * ScalarOps<T>::cosh(x.im),
        -ScalarOps<T>::sin(x.re) * ScalarOps<T>::sinh(x.im)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_tan(TfComplex<T> x) {
    return complex_div(complex_sin(x), complex_cos(x));
}

template <typename T>
__device__ inline TfComplex<T> complex_sinh(TfComplex<T> x) {
    return tf_complex<T>(
        ScalarOps<T>::sinh(x.re) * ScalarOps<T>::cos(x.im),
        ScalarOps<T>::cosh(x.re) * ScalarOps<T>::sin(x.im)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_cosh(TfComplex<T> x) {
    return tf_complex<T>(
        ScalarOps<T>::cosh(x.re) * ScalarOps<T>::cos(x.im),
        ScalarOps<T>::sinh(x.re) * ScalarOps<T>::sin(x.im)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_tanh(TfComplex<T> x) {
    return complex_div(complex_sinh(x), complex_cosh(x));
}

template <typename T>
__device__ inline TfComplex<T> complex_asinh(TfComplex<T> x) {
    return complex_log(
        complex_add(
            x,
            complex_sqrt(complex_add(complex_mul(x, x), complex_one<T>()))
        )
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_acosh(TfComplex<T> x) {
    return complex_log(
        complex_add(
            x,
            complex_mul(
                complex_sqrt(complex_add(x, complex_one<T>())),
                complex_sqrt(complex_sub(x, complex_one<T>()))
            )
        )
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_atanh(TfComplex<T> x) {
    return complex_scale(
        complex_sub(
            complex_log(complex_add(complex_one<T>(), x)),
            complex_log(complex_sub(complex_one<T>(), x))
        ),
        ScalarOps<T>::half()
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_asin(TfComplex<T> x) {
    return complex_mul_neg_i(complex_asinh(complex_mul_i(x)));
}

template <typename T>
__device__ inline TfComplex<T> complex_acos(TfComplex<T> x) {
    return complex_sub(
        complex_from_real<T>(ScalarOps<T>::pi() * ScalarOps<T>::half()),
        complex_asin(x)
    );
}

template <typename T>
__device__ inline TfComplex<T> complex_atan(TfComplex<T> x) {
    return complex_mul_neg_i(complex_atanh(complex_mul_i(x)));
}

template <typename T>
__device__ inline TfComplex<T> complex_pow(TfComplex<T> x, TfComplex<T> y) {
    return complex_exp(complex_mul(y, complex_log(x)));
}

template <typename T>
__device__ inline TfComplex<T> apply_unary(int op, TfComplex<T> x) {
    switch (op) {
        case UNARY_NEG:
            return tf_complex<T>(-x.re, -x.im);
        case UNARY_CONJ:
            return complex_conj(x);
        case UNARY_ABS:
            return complex_from_real<T>(complex_abs_value(x));
        case UNARY_RECIPROCAL:
            return complex_inv(x);
        case UNARY_REAL:
            return complex_from_real<T>(x.re);
        case UNARY_IMAG:
            return complex_from_real<T>(x.im);
        case UNARY_SQRT:
            return complex_sqrt(x);
        case UNARY_RSQRT:
            return complex_inv(complex_sqrt(x));
        case UNARY_EXP:
            return complex_exp(x);
        case UNARY_EXPM1:
            return complex_sub(complex_exp(x), complex_one<T>());
        case UNARY_LOG:
            return complex_log(x);
        case UNARY_LOG1P:
            return complex_log(complex_add(x, complex_one<T>()));
        case UNARY_SIN:
            return complex_sin(x);
        case UNARY_COS:
            return complex_cos(x);
        case UNARY_TAN:
            return complex_tan(x);
        case UNARY_TANH:
            return complex_tanh(x);
        case UNARY_ASIN:
            return complex_asin(x);
        case UNARY_ACOS:
            return complex_acos(x);
        case UNARY_ATAN:
            return complex_atan(x);
        case UNARY_SINH:
            return complex_sinh(x);
        case UNARY_COSH:
            return complex_cosh(x);
        case UNARY_ASINH:
            return complex_asinh(x);
        case UNARY_ACOSH:
            return complex_acosh(x);
        case UNARY_ATANH:
            return complex_atanh(x);
        default:
            return complex_zero<T>();
    }
}

template <typename T>
__device__ inline TfComplex<T> apply_binary(int op, TfComplex<T> x, TfComplex<T> y) {
    switch (op) {
        case BINARY_SUB:
            return complex_sub(x, y);
        case BINARY_DIV:
            return complex_div(x, y);
        case BINARY_POW:
            return complex_pow(x, y);
        case BINARY_XLOGY:
            return complex_is_zero(x) ? complex_zero<T>() : complex_mul(x, complex_log(y));
        default:
            return complex_zero<T>();
    }
}

extern "C" {

__global__ void tf_pointwise_unary_c32(
    const TfComplex<float>* input,
    TfComplex<float>* output,
    unsigned long long numel,
    int op,
    TfComplex<float> alpha,
    TfComplex<float> beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    TfComplex<float> current = output[idx];
    TfComplex<float> result = apply_unary(op, input[idx]);
    output[idx] = complex_add(complex_mul(alpha, result), complex_mul(beta, current));
}

__global__ void tf_pointwise_unary_c64(
    const TfComplex<double>* input,
    TfComplex<double>* output,
    unsigned long long numel,
    int op,
    TfComplex<double> alpha,
    TfComplex<double> beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    TfComplex<double> current = output[idx];
    TfComplex<double> result = apply_unary(op, input[idx]);
    output[idx] = complex_add(complex_mul(alpha, result), complex_mul(beta, current));
}

__global__ void tf_pointwise_binary_c32(
    const TfComplex<float>* lhs,
    const TfComplex<float>* rhs,
    TfComplex<float>* output,
    unsigned long long numel,
    int op,
    TfComplex<float> alpha,
    TfComplex<float> beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    TfComplex<float> current = output[idx];
    TfComplex<float> result = apply_binary(op, lhs[idx], rhs[idx]);
    output[idx] = complex_add(complex_mul(alpha, result), complex_mul(beta, current));
}

__global__ void tf_pointwise_binary_c64(
    const TfComplex<double>* lhs,
    const TfComplex<double>* rhs,
    TfComplex<double>* output,
    unsigned long long numel,
    int op,
    TfComplex<double> alpha,
    TfComplex<double> beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    TfComplex<double> current = output[idx];
    TfComplex<double> result = apply_binary(op, lhs[idx], rhs[idx]);
    output[idx] = complex_add(complex_mul(alpha, result), complex_mul(beta, current));
}

}
