extern "C" {

enum UnaryOp : int {
    UNARY_IDENTITY = 0,
    UNARY_IMAG_ZERO = 1,
    UNARY_EXPM1 = 2,
    UNARY_LOG1P = 3,
    UNARY_RSQRT = 4
};

enum BinaryOp : int {
    BINARY_POW = 0,
    BINARY_ATAN2 = 1,
    BINARY_HYPOT = 2,
    BINARY_XLOGY = 3
};

__device__ inline float apply_unary_f32(int op, float x) {
    switch (op) {
        case UNARY_IDENTITY:
            return x;
        case UNARY_IMAG_ZERO:
            return 0.0f;
        case UNARY_EXPM1:
            return expm1f(x);
        case UNARY_LOG1P:
            return log1pf(x);
        case UNARY_RSQRT:
            return 1.0f / sqrtf(x);
        default:
            return 0.0f;
    }
}

__device__ inline double apply_unary_f64(int op, double x) {
    switch (op) {
        case UNARY_IDENTITY:
            return x;
        case UNARY_IMAG_ZERO:
            return 0.0;
        case UNARY_EXPM1:
            return expm1(x);
        case UNARY_LOG1P:
            return log1p(x);
        case UNARY_RSQRT:
            return 1.0 / sqrt(x);
        default:
            return 0.0;
    }
}

__device__ inline float apply_binary_f32(int op, float x, float y) {
    switch (op) {
        case BINARY_POW:
            return powf(x, y);
        case BINARY_ATAN2:
            return atan2f(x, y);
        case BINARY_HYPOT:
            return hypotf(x, y);
        case BINARY_XLOGY:
            return x == 0.0f ? 0.0f : x * logf(y);
        default:
            return 0.0f;
    }
}

__device__ inline double apply_binary_f64(int op, double x, double y) {
    switch (op) {
        case BINARY_POW:
            return pow(x, y);
        case BINARY_ATAN2:
            return atan2(x, y);
        case BINARY_HYPOT:
            return hypot(x, y);
        case BINARY_XLOGY:
            return x == 0.0 ? 0.0 : x * log(y);
        default:
            return 0.0;
    }
}

__global__ void tf_pointwise_unary_f32(
    const float* input,
    float* output,
    unsigned long long numel,
    int op,
    float alpha,
    float beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float current = output[idx];
    output[idx] = alpha * apply_unary_f32(op, input[idx]) + beta * current;
}

__global__ void tf_pointwise_unary_f64(
    const double* input,
    double* output,
    unsigned long long numel,
    int op,
    double alpha,
    double beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    double current = output[idx];
    output[idx] = alpha * apply_unary_f64(op, input[idx]) + beta * current;
}

__global__ void tf_pointwise_binary_f32(
    const float* lhs,
    const float* rhs,
    float* output,
    unsigned long long numel,
    int op,
    float alpha,
    float beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    float current = output[idx];
    output[idx] = alpha * apply_binary_f32(op, lhs[idx], rhs[idx]) + beta * current;
}

__global__ void tf_pointwise_binary_f64(
    const double* lhs,
    const double* rhs,
    double* output,
    unsigned long long numel,
    int op,
    double alpha,
    double beta
) {
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numel) {
        return;
    }
    double current = output[idx];
    output[idx] = alpha * apply_binary_f64(op, lhs[idx], rhs[idx]) + beta * current;
}

}
