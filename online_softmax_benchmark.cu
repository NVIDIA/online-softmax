#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cub/cub.cuh>
#include <curand.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

#define CUDA_CHECK(callstr) {cudaError_t error_code = callstr; if (error_code != cudaSuccess) { std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__; assert(0); } }
#define CURAND_CHECK(callstr) {curandStatus_t error_code = callstr; if (error_code != CURAND_STATUS_SUCCESS) { std::cerr << "cuRAND error " << error_code << " at " << __FILE__ << ":" << __LINE__; assert(0); } }

const int MAX_K=5;

enum SOFTMAX_TYPE
{
    SOFTMAX_TYPE_NAIVE,
    SOFTMAX_TYPE_SAFE,
    SOFTMAX_TYPE_ONLINE
};

enum SOFTMAX_TOPK_TYPE
{
    SOFTMAX_TOPK_TYPE_TOPK_ONLY,
    SOFTMAX_TOPK_TYPE_SAFE_UNFUSED,
    SOFTMAX_TOPK_TYPE_SAFE_FUSED,
    SOFTMAX_TOPK_TYPE_ONLINE_FUSED
};

std::string getSoftmaxTypeName(SOFTMAX_TYPE t)
{
    switch (t)
    {
    case SOFTMAX_TYPE_NAIVE:
        return "Naive Softmax";
    case SOFTMAX_TYPE_SAFE:
        return "Safe Softmax";
    case SOFTMAX_TYPE_ONLINE:
        return "Online Softmax";
    default:
        assert(0);
        break;
    }
    return "";
}

std::string getSoftmaxTopkTypeName(SOFTMAX_TOPK_TYPE t)
{
    switch (t)
    {
    case SOFTMAX_TOPK_TYPE_TOPK_ONLY:
        return "TopK";
    case SOFTMAX_TOPK_TYPE_SAFE_UNFUSED:
        return "Safe Softmax + TopK unfused";
    case SOFTMAX_TOPK_TYPE_SAFE_FUSED:
        return "Safe Softmax + TopK fused";
    case SOFTMAX_TOPK_TYPE_ONLINE_FUSED:
        return "Online Softmax + TopK fused";
    default:
        assert(0);
        break;
    }
    return "";
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void naive_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float d_total_inverse;

    float d_partial = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        d_partial += __expf(x[elem_id]);

    float d = BlockReduce(temp_storage).Sum(d_partial);
    if (thread_id == 0)
        d_total_inverse = __fdividef(1.0F, d);
    __syncthreads();

    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id]) * d_total_inverse;
}

__device__ __forceinline__ float max_op(float a, float b)
{
    return fmaxf(a, b);
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void safe_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float m_total;
    __shared__ float d_total_inverse;

    float m_partial = -FLT_MAX;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        m_partial = max_op(m_partial, x[elem_id]);

    float m = BlockReduce(temp_storage).Reduce(m_partial, max_op);
    if (thread_id == 0)
        m_total = m;
    __syncthreads();

    float d_partial = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        d_partial += __expf(x[elem_id] - m_total);

    float d = BlockReduce(temp_storage).Sum(d_partial);
    if (thread_id == 0)
        d_total_inverse = __fdividef(1.0F, d);
    __syncthreads();

    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - m_total) * d_total_inverse;
}

struct __align__(8) MD
{
    float m;
    float d;
};

__device__ __forceinline__ MD reduce_md_op(MD a, MD b)
{
    bool a_bigger = (a.m > b.m);
    MD bigger_m = a_bigger ? a : b;
    MD smaller_m = a_bigger ? b : a;
    MD res;
    res.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
    res.m = bigger_m.m;
    return res;
}

template<int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax(
    const float * __restrict x,
    float * __restrict y,
    int V)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition x and y to data for the current vector
    x += vector_id * V;
    y += vector_id * V;

    typedef cub::BlockReduce<MD, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ MD md_total;

    MD md_partial;
    md_partial.m = -FLT_MAX;
    md_partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        MD new_elem;
        new_elem.m = x[elem_id];
        new_elem.d = 1.0F;
        md_partial = reduce_md_op(md_partial, new_elem);
    }

    MD md = BlockReduce(temp_storage).Reduce(md_partial, reduce_md_op);
    if (thread_id == 0)
        md_total = md;
    __syncthreads();

    float d_total_inverse = __fdividef(1.0F, md_total.d);
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        y[elem_id] = __expf(x[elem_id] - md_total.m) * d_total_inverse;
}

template<int MAX_K>
struct TopK
{
    int p[MAX_K];
    float u[MAX_K];

    __device__ __forceinline__ void insert(float elem, int elem_id)
    {
        if (elem > u[MAX_K-1])
        {
            u[MAX_K-1] = elem;
            p[MAX_K-1] = elem_id;
        }
        for(int k = MAX_K - 2; k >= 0; --k)
        {
            if (u[k+1] > u[k])
            {
                float u2 = u[k];
                int p2 = p[k];
                u[k] = u[k+1];
                p[k] = p[k+1];
                u[k+1] = u2;
                p[k+1] = p2;
            }
        }
    }
};

template<int MAX_K>
__device__ __forceinline__ TopK<MAX_K> reduce_topk_op(const TopK<MAX_K>& a, const TopK<MAX_K>& b)
{
    TopK<MAX_K> res = a;
    for(int i = 0; i < MAX_K; ++i)
        res.insert(b.u[i], b.p[i]);
    return res;
}

template<int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void topk(
    const float * __restrict y,
    int * __restrict z,
    float * __restrict v,
    int V,
    int K)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition y to data for the current vector
    y += vector_id * V;

    typedef cub::BlockReduce<TopK<MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopK<MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
        partial.p[i] = -1;
    for(int i = 0; i < MAX_K; ++i)
        partial.u[i] = -FLT_MAX;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        float elem = y[elem_id];
        partial.insert(elem, elem_id);
    }

    TopK<MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        
        for(int i = 0; i < MAX_K; ++i)
        {
            if (i < K)
            {
                z[i] = total.p[i];
                v[i] = total.u[i];
            }
        }
    }
}

template<int MAX_K>
struct TopKD
{
    float d;
    TopK<MAX_K> topk;
};

template<int MAX_K>
__device__ __forceinline__ TopKD<MAX_K> reduce_topk_d_op(const TopKD<MAX_K>& a, const TopKD<MAX_K>& b)
{
    TopKD<MAX_K> res;
    res.d = a.d + b.d;
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template<int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void safe_softmax_topk(
    const float * __restrict x,
    int * __restrict z,
    float * __restrict v,
    int V,
    int K)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<float, THREADBLOCK_SIZE> MaxValBlockReduce;
    typedef cub::BlockReduce<TopKD<MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename MaxValBlockReduce::TempStorage max_val_temp_storage;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float m_total;

    float m_partial = -FLT_MAX;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
        m_partial = max_op(m_partial, x[elem_id]);

    float m = MaxValBlockReduce(max_val_temp_storage).Reduce(m_partial, max_op);
    if (thread_id == 0)
        m_total = m;
    __syncthreads();

    TopKD<MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
        partial.topk.p[i] = -1;
    for(int i = 0; i < MAX_K; ++i)
        partial.topk.u[i] = -FLT_MAX;
    partial.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        float elem = x[elem_id];
        partial.d += __expf(elem - m_total);
        partial.topk.insert(elem, elem_id);
    }

    TopKD<MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_d_op<MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        
        float d_total_inverse = __fdividef(1.0F, total.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = __expf(total.topk.u[i] - m_total) * d_total_inverse;
            if (i < K)
            {
                z[i] = total.topk.p[i];
                v[i] = val;
            }
        }
    }
}

template<int MAX_K>
struct TopKMD
{
    MD md;
    TopK<MAX_K> topk;
};

template<int MAX_K>
__device__ __forceinline__ TopKMD<MAX_K> reduce_topk_md_op(const TopKMD<MAX_K>& a, const TopKMD<MAX_K>& b)
{
    TopKMD<MAX_K> res;
    res.md = reduce_md_op(a.md, b.md);
    res.topk = reduce_topk_op(a.topk, b.topk);
    return res;
}

template<int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE)
__global__ void online_softmax_topk(
    const float * __restrict x,
    int * __restrict z,
    float * __restrict v,
    int V,
    int K)
{
    int thread_id = threadIdx.x;
    int vector_id = blockIdx.x;

    // reposition y to data for the current vector
    x += vector_id * V;

    typedef cub::BlockReduce<TopKMD<MAX_K>, THREADBLOCK_SIZE> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    TopKMD<MAX_K> partial;
    for(int i = 0; i < MAX_K; ++i)
        partial.topk.p[i] = -1;
    for(int i = 0; i < MAX_K; ++i)
        partial.topk.u[i] = -FLT_MAX;
    partial.md.m = -FLT_MAX;
    partial.md.d = 0.0F;
    for(int elem_id = thread_id; elem_id < V; elem_id += THREADBLOCK_SIZE)
    {
        float elem = x[elem_id];
        MD new_elem{elem, 1.0F};
        partial.md = reduce_md_op(partial.md, new_elem);
        partial.topk.insert(elem, elem_id);
    }

    TopKMD<MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_md_op<MAX_K>);

    if (thread_id == 0)
    {
        z += vector_id * K;
        v += vector_id * K;
        
        float d_total_inverse = __fdividef(1.0F, total.md.d);
        for(int i = 0; i < MAX_K; ++i)
        {
            float val = __expf(total.topk.u[i] - total.md.m) * d_total_inverse;
            if (i < K)
            {
                z[i] = total.topk.p[i];
                v[i] = val;
            }
        }
    }
}

void fill_random_values(float * x, int count)
{
    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
    CURAND_CHECK(curandGenerateUniform(gen, x, count));
    CURAND_CHECK(curandDestroyGenerator(gen));
}

std::vector<float> run_softmax(int V, int batch_size, SOFTMAX_TYPE t)
{
    float * x;
    float * y;
    CUDA_CHECK(cudaMalloc(&x, (size_t)V * batch_size * sizeof(float)));
    fill_random_values(x, V * batch_size);
    CUDA_CHECK(cudaMalloc(&y, (size_t)V * batch_size * sizeof(float)));

    switch (t)
    {
    case SOFTMAX_TYPE_NAIVE:
        naive_softmax<256><<<batch_size,256>>>(x, y, V);
        break;
    case SOFTMAX_TYPE_SAFE:
        safe_softmax<256><<<batch_size,256>>>(x, y, V);
        break;
    case SOFTMAX_TYPE_ONLINE:
        online_softmax<256><<<batch_size,256>>>(x, y, V);
        break;
    default:
        assert(0);
    }

    std::vector<float> res(V * batch_size);
    CUDA_CHECK(cudaMemcpy(&res[0], y, V * batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    return res;
}

void compare_softmax_results(int V, int batch_size, SOFTMAX_TYPE t1, SOFTMAX_TYPE t2)
{
    std::vector<float> res1 = run_softmax(V, batch_size, t1);
    std::vector<float> res2 = run_softmax(V, batch_size, t2);

    float max_diff = 0.0F;
    double total_diff = 0.0F;
    for(int i = 0; i < res1.size(); ++i)
    {
        float diff = fabs(res1[i] - res2[i]);
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }
    std::cout << "Comparing " << getSoftmaxTypeName(t1) << " and " << getSoftmaxTypeName(t2)
        << ": Max diff = " << max_diff << ", Avg diff = " << (float)(total_diff / res1.size()) << std::endl;
}

// Returns runtime, in seconds
float benchmark_softmax(int V, int batch_size, SOFTMAX_TYPE t, int run_iterations)
{
    float * x;
    float * y;
    CUDA_CHECK(cudaMalloc(&x, (size_t)V * batch_size * sizeof(float)));
    fill_random_values(x, V * batch_size);
    CUDA_CHECK(cudaMalloc(&y, (size_t)V * batch_size * sizeof(float)));

    // Heuristic to have at least 8 iterations of the loop
    int max_threadblock_size = V / 8;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for(int i = 0; i < run_iterations; ++i)
    {
        switch (t)
        {
        case SOFTMAX_TYPE_NAIVE:
            if (max_threadblock_size >= 256)
                naive_softmax<256><<<batch_size,256>>>(x, y, V);
            else if (max_threadblock_size >= 128)
                naive_softmax<128><<<batch_size,128>>>(x, y, V);
            else if (max_threadblock_size >= 64)
                naive_softmax<64><<<batch_size,64>>>(x, y, V);
            else
                naive_softmax<32><<<batch_size,32>>>(x, y, V);
            break;
        case SOFTMAX_TYPE_SAFE:
            if (max_threadblock_size >= 256)
                safe_softmax<256><<<batch_size,256>>>(x, y, V);
            else if (max_threadblock_size >= 128)
                safe_softmax<128><<<batch_size,128>>>(x, y, V);
            else if (max_threadblock_size >= 64)
                safe_softmax<64><<<batch_size,64>>>(x, y, V);
            else
                safe_softmax<32><<<batch_size,32>>>(x, y, V);
            break;
        case SOFTMAX_TYPE_ONLINE:
            if (max_threadblock_size >= 256)
                online_softmax<256><<<batch_size,256>>>(x, y, V);
            else if (max_threadblock_size >= 128)
                online_softmax<128><<<batch_size,128>>>(x, y, V);
            else if (max_threadblock_size >= 64)
                online_softmax<64><<<batch_size,64>>>(x, y, V);
            else
                online_softmax<32><<<batch_size,32>>>(x, y, V);
            break;
        default:
            assert(0);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    
    return elapsedTime / run_iterations * 0.001F;
}

// Returns runtime, in seconds
float benchmark_softmax_topk(int V, int K, int batch_size, SOFTMAX_TOPK_TYPE t, int run_iterations)
{
    assert(K<=MAX_K);

    float * x;
    float * y;
    int * z;
    float * v;
    CUDA_CHECK(cudaMalloc(&x, (size_t)V * batch_size * sizeof(float)));
    fill_random_values(x, V * batch_size);
    CUDA_CHECK(cudaMalloc(&y, (size_t)V * batch_size * sizeof(float)));
    fill_random_values(y, V * batch_size);
    CUDA_CHECK(cudaMalloc(&z, (size_t)K * batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&v, (size_t)K * batch_size * sizeof(float)));

    // Heuristic to have at least 16 iterations of the loop
    int max_threadblock_size = V / 16;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, 0));
    for(int i = 0; i < run_iterations; ++i)
    {
        switch (t)
        {
        case SOFTMAX_TOPK_TYPE_TOPK_ONLY:
            if (max_threadblock_size >= 256)
                topk<MAX_K,256><<<batch_size,256>>>(y, z, v, V, K);
            else if (max_threadblock_size >= 128)
                topk<MAX_K,128><<<batch_size,128>>>(y, z, v, V, K);
            else if (max_threadblock_size >= 64)
                topk<MAX_K,64><<<batch_size,64>>>(y, z, v, V, K);
            else
                topk<MAX_K,32><<<batch_size,32>>>(y, z, v, V, K);
            break;
        case SOFTMAX_TOPK_TYPE_SAFE_UNFUSED:
            if (max_threadblock_size >= 256)
            {
                safe_softmax<256><<<batch_size,256>>>(x, y, V);
                topk<MAX_K,256><<<batch_size,256>>>(y, z, v, V, K);
            }
            else if (max_threadblock_size >= 128)
            {
                safe_softmax<128><<<batch_size,128>>>(x, y, V);
                topk<MAX_K,128><<<batch_size,128>>>(y, z, v, V, K);
            }
            else if (max_threadblock_size >= 64)
            {
                safe_softmax<64><<<batch_size,64>>>(x, y, V);
                topk<MAX_K,64><<<batch_size,64>>>(y, z, v, V, K);
            }
            else
            {
                safe_softmax<32><<<batch_size,32>>>(x, y, V);
                topk<MAX_K,32><<<batch_size,32>>>(y, z, v, V, K);
            }
            break;
        case SOFTMAX_TOPK_TYPE_SAFE_FUSED:
            if (max_threadblock_size >= 256)
                safe_softmax_topk<MAX_K,256><<<batch_size,256>>>(x, z, v, V, K);
            else if (max_threadblock_size >= 128)
                safe_softmax_topk<MAX_K,128><<<batch_size,128>>>(x, z, v, V, K);
            else if (max_threadblock_size >= 64)
                safe_softmax_topk<MAX_K,64><<<batch_size,64>>>(x, z, v, V, K);
            else
                safe_softmax_topk<MAX_K,32><<<batch_size,32>>>(x, z, v, V, K);
            break;
        case SOFTMAX_TOPK_TYPE_ONLINE_FUSED:
            if (max_threadblock_size >= 256)
                online_softmax_topk<MAX_K,256><<<batch_size,256>>>(x, z, v, V, K);
            else if (max_threadblock_size >= 128)
                online_softmax_topk<MAX_K,128><<<batch_size,128>>>(x, z, v, V, K);
            else if (max_threadblock_size >= 64)
                online_softmax_topk<MAX_K,64><<<batch_size,64>>>(x, z, v, V, K);
            else
                online_softmax_topk<MAX_K,32><<<batch_size,32>>>(x, z, v, V, K);
            break;
        default:
            assert(0);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(x));
    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaFree(v));
    
    return elapsedTime / run_iterations * 0.001F;
}

std::tuple<std::vector<float>,std::vector<int>,std::vector<float>> run_topk(int V, int K, int batch_size, SOFTMAX_TOPK_TYPE t)
{
    assert(K<=MAX_K);

    float * y;
    int * z;
    float * v;
    CUDA_CHECK(cudaMalloc(&y, (size_t)V * batch_size * sizeof(float)));
    fill_random_values(y, V * batch_size);
    CUDA_CHECK(cudaMalloc(&z, (size_t)K * batch_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&v, (size_t)K * batch_size * sizeof(float)));

    switch (t)
    {
    case SOFTMAX_TOPK_TYPE_TOPK_ONLY:
        topk<MAX_K,256><<<batch_size,256>>>(y, z, v, V, K);
        break;
    case SOFTMAX_TOPK_TYPE_SAFE_FUSED:
        safe_softmax_topk<MAX_K,256><<<batch_size,256>>>(y, z, v, V, K);
        break;
    case SOFTMAX_TOPK_TYPE_ONLINE_FUSED:
        online_softmax_topk<MAX_K,256><<<batch_size,256>>>(y, z, v, V, K);
        break;
    default:
        assert(0);
    }

    std::vector<float> yh(V * batch_size);
    std::vector<int> zh(K * batch_size);
    std::vector<float> vh(K * batch_size);

    CUDA_CHECK(cudaMemcpy(&yh[0], y, (size_t)V * batch_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&zh[0], z, (size_t)K * batch_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&vh[0], v, (size_t)K * batch_size * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(y));
    CUDA_CHECK(cudaFree(z));
    CUDA_CHECK(cudaFree(v));
    
    return std::make_tuple(yh, zh, vh);
}

void compare_topk_results(int V, int K, int batch_size, SOFTMAX_TOPK_TYPE t)
{
    std::vector<float> yh;
    std::vector<int> zh;
    std::vector<float> vh;
    
    std::tie(yh, zh, vh) = run_topk(V, K, batch_size, t);

    auto y = yh.begin();
    auto z = zh.begin();
    auto v = vh.begin();
    int mismatches = 0;
    for(int i = 0; i < batch_size; ++i, y += V, z += K, v += K)
    {
        std::vector<std::pair<float,int>> elemsWithIndices;
        for(int j = 0; j < V; ++j)
            elemsWithIndices.push_back(std::make_pair(*(y+j), j));
        std::partial_sort(elemsWithIndices.begin(), elemsWithIndices.begin() + K, elemsWithIndices.end(),
            [] (const std::pair<float,int>& a, const std::pair<float,int>& b) { if (a.first > b.first) return true; if (a.first < b.first) return false; return a.second < b.second; });
        for(int j = 0; j < K; ++j)
        {
            if ((*(z+j) != elemsWithIndices[j].second) || (*(v+j) != elemsWithIndices[j].first))
            {
                std::cout << getSoftmaxTopkTypeName(t) << " mismatch for vector " << i << ", reference (" << elemsWithIndices[j].second << "," << elemsWithIndices[j].first
                    << "), GPU (" << *(z+j) << "," << *(v+j) << ")" << std::endl;
                ++mismatches;
            }
        }
    }
    std::cout << getSoftmaxTopkTypeName(t) << ": " << mismatches << " mismatches" << std::endl;
}

void compare_softmax_topk_results(int V, int K, int batch_size, SOFTMAX_TOPK_TYPE t)
{
    std::vector<float> xh;
    std::vector<int> zh;
    std::vector<float> vh;
    
    std::tie(xh, zh, vh) = run_topk(V, K, batch_size, t);

    auto x = xh.begin();
    auto z = zh.begin();
    auto v = vh.begin();
    int mismatches = 0;
    float max_diff = 0.0F;
    double total_diff = 0.0F;
    for(int i = 0; i < batch_size; ++i, x += V, z += K, v += K)
    {
        // Compute reference softmax
        float m = 0.0F;
        for(int j = 0; j < V; ++j)
            m = std::max(m, *(x+j));
        float d = 0.0F;
        for(int j = 0; j < V; ++j)
            d += expf(*(x+j) - m);
        for(int j = 0; j < V; ++j)
            *(x+j) = expf(*(x+j) - m) / d;

        std::vector<std::pair<float,int>> elemsWithIndices;
        for(int j = 0; j < V; ++j)
            elemsWithIndices.push_back(std::make_pair(*(x+j), j));
        std::partial_sort(elemsWithIndices.begin(), elemsWithIndices.begin() + K, elemsWithIndices.end(),
            [] (const std::pair<float,int>& a, const std::pair<float,int>& b) { if (a.first > b.first) return true; if (a.first < b.first) return false; return a.second < b.second; });
        for(int j = 0; j < K; ++j)
        {
            float diff = fabs(*(v+j) - elemsWithIndices[j].first);
            max_diff = std::max(max_diff, diff);
            total_diff += diff;
            if (*(z+j) != elemsWithIndices[j].second)
            {
                std::cout << getSoftmaxTopkTypeName(t) << " mismatch for vector " << i << ", reference (" << elemsWithIndices[j].second << "," << elemsWithIndices[j].first
                    << "), GPU (" << *(z+j) << "," << *(v+j) << ")" << std::endl;
                ++mismatches;
            }
        }
    }
    std::cout << getSoftmaxTopkTypeName(t) << ": " << mismatches << " mismatches, comparing to CPU reference implementation: Max diff = " << max_diff << ", Avg diff = " << (float)(total_diff / (batch_size * K)) << std::endl;
}

void run_benchmark(int batch_size, int start_V, int K, int end_V, int average_run_iterations, int min_run_iteration)
{
    std::cout << "Batch size = " << batch_size << std::endl;
    std::cout << std::setw(12) << "V";
    std::cout << std::setw(20) << "NaiveSoftmax";
    std::cout << std::setw(20) << "SafeSoftmax";
    std::cout << std::setw(20) << "OnlineSoftmax";
    std::cout << std::setw(20) << "TopK";
    std::cout << std::setw(30) << "SafeSoftmaxUnfusedTopK";
    std::cout << std::setw(30) << "SafeSoftmaxFusedTopK";
    std::cout << std::setw(30) << "OnlineSoftmaxFusedTopK";
    std::cout << std::endl;
    float average_V = sqrtf(static_cast<float>(end_V)*static_cast<float>(start_V));
    for(int V = start_V; V < end_V; V *= 2)
    {
        int run_iterations = std::max(static_cast<int>(static_cast<float>(average_run_iterations) * average_V / static_cast<float>(V)), min_run_iteration);
        std::cout << std::setw(12) << V;
        {
            float runtime = benchmark_softmax(V, batch_size, SOFTMAX_TYPE_NAIVE, run_iterations);
            std::cout << std::setw(20) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax(V, batch_size, SOFTMAX_TYPE_SAFE, run_iterations);
            std::cout << std::setw(20) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax(V, batch_size, SOFTMAX_TYPE_ONLINE, run_iterations);
            std::cout << std::setw(20) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax_topk(V, K, batch_size, SOFTMAX_TOPK_TYPE_TOPK_ONLY, run_iterations);
            std::cout << std::setw(20) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax_topk(V, K, batch_size, SOFTMAX_TOPK_TYPE_SAFE_UNFUSED, run_iterations);
            std::cout << std::setw(30) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax_topk(V, K, batch_size, SOFTMAX_TOPK_TYPE_SAFE_FUSED, run_iterations);
            std::cout << std::setw(30) << (V * batch_size / runtime);
        }
        {
            float runtime = benchmark_softmax_topk(V, K, batch_size, SOFTMAX_TOPK_TYPE_ONLINE_FUSED, run_iterations);
            std::cout << std::setw(30) << (V * batch_size / runtime);
        }

        std::cout << std::endl;
    }
}

int main(int argc, char *argv[])
{
    std::cout << "Softmax correctness check:" << std::endl;
    compare_softmax_results(300, 100, SOFTMAX_TYPE_NAIVE, SOFTMAX_TYPE_SAFE);
    compare_softmax_results(300, 100, SOFTMAX_TYPE_NAIVE, SOFTMAX_TYPE_ONLINE);
    std::cout << "TopK correctness check:" << std::endl;
    compare_topk_results(300, MAX_K, 100, SOFTMAX_TOPK_TYPE_TOPK_ONLY);
    std::cout << "Softmax+TopK correctness check:" << std::endl;
    compare_softmax_topk_results(300, MAX_K, 100, SOFTMAX_TOPK_TYPE_SAFE_FUSED);
    compare_softmax_topk_results(300, MAX_K, 100, SOFTMAX_TOPK_TYPE_ONLINE_FUSED);
    
    int large_batch_size = 4000;
    int small_batch_size = 10;
    size_t max_V = 10000000;

    int start_V = 63;
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
    int large_batch_end_V = std::min(static_cast<size_t>(device_prop.totalGlobalMem * 0.9F) / (sizeof(float) * 3 * large_batch_size), max_V);
    int small_batch_end_V = std::min(static_cast<size_t>(device_prop.totalGlobalMem * 0.9F) / (sizeof(float) * 3 * small_batch_size), max_V);

    std::cout << "Softmax benchmark:" << std::endl;
    run_benchmark(large_batch_size, start_V, MAX_K, large_batch_end_V, 100, 10);
    run_benchmark(small_batch_size, start_V, MAX_K, small_batch_end_V, 4000, 800);

    return 0;
}
