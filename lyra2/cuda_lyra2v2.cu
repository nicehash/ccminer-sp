

#include <stdio.h>
#include <memory.h>
#include "cuda_vector.h"
#define TPB52 256
#define TPB50 64


#define Nrow 4
#define Ncol 4
#define u64type uint4
#define vectype uint48
#define memshift 3

__device__ uint28 *DState;

__device__ __forceinline__ uint28 LD8S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	uint28 ret;
	ret.x = shared_mem[(index * 4 + 0) * blockDim.x + threadIdx.x];
	ret.y = shared_mem[(index * 4 + 1) * blockDim.x + threadIdx.x];
	ret.z = shared_mem[(index * 4 + 2) * blockDim.x + threadIdx.x];
	ret.w = shared_mem[(index * 4 + 3) * blockDim.x + threadIdx.x];

	return ret;
}

__device__ __forceinline__ void ST8S(const int index, const uint28 &data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * 4 + 0) * blockDim.x + threadIdx.x] = data.x;
	shared_mem[(index * 4 + 1) * blockDim.x + threadIdx.x] = data.y;
	shared_mem[(index * 4 + 2) * blockDim.x + threadIdx.x] = data.z;
	shared_mem[(index * 4 + 3) * blockDim.x + threadIdx.x] = data.w;
}

__device__ __forceinline__ uint2 LD4S(const int index)
{
	extern __shared__ uint2 shared_mem[];

	return shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x];
}

__device__ __forceinline__ void ST4S(const int index, const uint2 data)
{
	extern __shared__ uint2 shared_mem[];

	shared_mem[(index * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x] = data;
}

__device__ __forceinline__ void Gfunc_v35(uint2 &a, uint2 &b, uint2 &c, uint2 &d)
{

	a += b; d = eorswap32(a, d);
	c += d; b ^= c; b = ROR24(b);
	a += b; d ^= a; d = ROR16(d);
	c += d; b ^= c; b = ROR2(b, 63);

}

__device__ __forceinline__ void round_lyra_v35(uint28 s[4])
{
	Gfunc_v35(s[0].x, s[1].x, s[2].x, s[3].x);
	Gfunc_v35(s[0].y, s[1].y, s[2].y, s[3].y);
	Gfunc_v35(s[0].z, s[1].z, s[2].z, s[3].z);
	Gfunc_v35(s[0].w, s[1].w, s[2].w, s[3].w);

	Gfunc_v35(s[0].x, s[1].y, s[2].z, s[3].w);
	Gfunc_v35(s[0].y, s[1].z, s[2].w, s[3].x);
	Gfunc_v35(s[0].z, s[1].w, s[2].x, s[3].y);
	Gfunc_v35(s[0].w, s[1].x, s[2].y, s[3].z);
}

__device__ __forceinline__ uint2 __shfl(uint2 a, uint32_t b, uint32_t c)
{
	return make_uint2(__shfl(a.x, b, c), __shfl(a.y, b, c));
}

__device__ __forceinline__ void round_lyra_v35(uint2 s[4])
{
	Gfunc_v35(s[0], s[1], s[2], s[3]);
	s[1] = __shfl(s[1], threadIdx.x + 1, 4);
	s[2] = __shfl(s[2], threadIdx.x + 2, 4);
	s[3] = __shfl(s[3], threadIdx.x + 3, 4);
	Gfunc_v35(s[0], s[1], s[2], s[3]);
	s[1] = __shfl(s[1], threadIdx.x + 3, 4);
	s[2] = __shfl(s[2], threadIdx.x + 2, 4);
	s[3] = __shfl(s[3], threadIdx.x + 1, 4);
}

__device__ __forceinline__ void reduceDuplexRowSetupV2(uint2 state[4])
{
	int i, j;
	uint2 state1[Ncol][3], state0[Ncol][3], state2[3];

#if __CUDA_ARCH__ > 500
#pragma unroll
#endif
	for (int i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] = state[j];
		round_lyra_v35(state);
	}

	//#pragma unroll 4
	for (i = 0; i < Ncol; i++)
	{
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state0[i][j];

		round_lyra_v35(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] = state0[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state1[Ncol - i - 1][j] ^= state[j];
	}

	for (i = 0; i < Ncol; i++)
	{
		const uint32_t s0 = memshift * Ncol * 0 + i * memshift;
		const uint32_t s2 = memshift * Ncol * 2 + memshift * (Ncol - 1) - i*memshift;
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[i][j];

		round_lyra_v35(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] = state1[i][j];

#pragma unroll
		for (j = 0; j < 3; j++)
			state2[j] ^= state[j];

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state0[i][0] ^= Data2;
			state0[i][1] ^= Data0;
			state0[i][2] ^= Data1;
		}
		else
		{
			state0[i][0] ^= Data0;
			state0[i][1] ^= Data1;
			state0[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s0 + j, state0[i][j]);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[i][j] = state2[j];

	}

	for (i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = memshift * Ncol * 1 + i*memshift;
		const uint32_t s3 = memshift * Ncol * 3 + memshift * (Ncol - 1) - i*memshift;
#pragma unroll
		for (j = 0; j < 3; j++)
			state[j] ^= state1[i][j] + state0[Ncol - i - 1][j];

		round_lyra_v35(state);

#pragma unroll
		for (j = 0; j < 3; j++)
			state0[Ncol - i - 1][j] ^= state[j];
#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s3 + j, state0[Ncol - i - 1][j]);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state1[i][0] ^= Data2;
			state1[i][1] ^= Data0;
			state1[i][2] ^= Data1;
		}
		else
		{
			state1[i][0] ^= Data0;
			state1[i][1] ^= Data1;
			state1[i][2] ^= Data2;
		}

#pragma unroll
		for (j = 0; j < 3; j++)
			ST4S(s1 + j, state1[i][j]);


	}
}

__device__ void reduceDuplexRowtV2(const int rowIn, const int rowInOut, const int rowOut, uint2 state[4])
{
	uint2 state1[3], state2[3];
	const uint32_t ps1 = memshift * Ncol * rowIn;
	const uint32_t ps2 = memshift * Ncol * rowInOut;
	const uint32_t ps3 = memshift * Ncol * rowOut;

	for (int i = 0; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;
		const uint32_t s3 = ps3 + i*memshift;

#pragma unroll
		for (int j = 0; j < 3; j++)
			state1[j] = LD4S(s1 + j);

#pragma unroll
		for (int j = 0; j < 3; j++)
			state2[j] = LD4S(s2 + j);

#pragma unroll 
		for (int j = 0; j < 3; j++)
			state[j] ^= state1[j] + state2[j];

		round_lyra_v35(state);

		//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
		uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
		uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
		uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

		if (threadIdx.x == 0)
		{
			state2[0] ^= Data2;
			state2[1] ^= Data0;
			state2[2] ^= Data1;
		}
		else
		{
			state2[0] ^= Data0;
			state2[1] ^= Data1;
			state2[2] ^= Data2;
		}

#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s2 + j, state2[j]);
#pragma unroll
		for (int j = 0; j < 3; j++)
			ST4S(s3 + j, LD4S(s3 + j) ^ state[j]);
	}
}

__device__ void reduceDuplexRowtV2_4(const int rowInOut, uint2 state[4])
{
	const int rowIn = 2;
	const int rowOut = 3;

	int i, j;
	uint2 state2[3], state1[3], last[3];
	const uint32_t ps1 = memshift * Ncol * rowIn;
	const uint32_t ps2 = memshift * Ncol * rowInOut;
	const uint32_t ps3 = memshift * Ncol * rowOut;

#pragma unroll
	for (int j = 0; j < 3; j++)
		last[j] = LD4S(ps2 + j);

#pragma unroll 
	for (int j = 0; j < 3; j++)
		state[j] ^= LD4S(ps1 + j) + last[j];

	round_lyra_v35(state);

	//一個手前のスレッドからデータを貰う(同時に一個先のスレッドにデータを送る)
	uint2 Data0 = __shfl(state[0], threadIdx.x - 1, 4);
	uint2 Data1 = __shfl(state[1], threadIdx.x - 1, 4);
	uint2 Data2 = __shfl(state[2], threadIdx.x - 1, 4);

	if (threadIdx.x == 0)
	{
		last[0] ^= Data2;
		last[1] ^= Data0;
		last[2] ^= Data1;
	}
	else
	{
		last[0] ^= Data0;
		last[1] ^= Data1;
		last[2] ^= Data2;
	}

	if (rowInOut == rowOut)
	{
#pragma unroll 
		for (j = 0; j < 3; j++)
			last[j] ^= state[j];
	}

	for (i = 1; i < Ncol; i++)
	{
		const uint32_t s1 = ps1 + i*memshift;
		const uint32_t s2 = ps2 + i*memshift;

#pragma unroll 
		for (j = 0; j < 3; j++)
			state[j] ^= LD4S(s1 + j) + LD4S(s2 + j);

		round_lyra_v35(state);
	}

#pragma unroll
	for (int j = 0; j < 3; j++)
		state[j] ^= last[j];
}

__constant__ uint28 blake2b_IV[2] = {
	0xf3bcc908lu, 0x6a09e667lu,
	0x84caa73blu, 0xbb67ae85lu,
	0xfe94f82blu, 0x3c6ef372lu,
	0x5f1d36f1lu, 0xa54ff53alu,
	0xade682d1lu, 0x510e527flu,
	0x2b3e6c1flu, 0x9b05688clu,
	0xfb41bd6blu, 0x1f83d9ablu,
	0x137e2179lu, 0x5be0cd19lu
};

__constant__ uint28 Mask[2] = {
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000020lu, 0x00000000lu,
	0x00000001lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000004lu, 0x00000000lu,
	0x00000080lu, 0x00000000lu,
	0x00000000lu, 0x01000000lu
};

__global__
__launch_bounds__(32, 1)
void lyra2v2_gpu_hash_32_1(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0].x = state[1].x = __ldg(&outputHash[thread + threads * 0]);
		state[0].y = state[1].y = __ldg(&outputHash[thread + threads * 1]);
		state[0].z = state[1].z = __ldg(&outputHash[thread + threads * 2]);
		state[0].w = state[1].w = __ldg(&outputHash[thread + threads * 3]);
		state[2] = blake2b_IV[0];
		state[3] = blake2b_IV[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		state[0] ^= Mask[0];
		state[1] ^= Mask[1];

		for (int i = 0; i<12; i++)
			round_lyra_v35(state);

		DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x] = state[0];
		DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x] = state[1];
		DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x] = state[2];
		DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x] = state[3];

	} //thread
}

__global__
__launch_bounds__(32, 1)
void lyra2v2_gpu_hash_32_2(uint32_t threads, uint32_t startNounce, uint64_t *outputHash)
{
	const uint32_t thread = blockDim.y * blockIdx.x + threadIdx.y;

	if (thread < threads)
	{
		uint2 state[4];
		state[0] = ((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[1] = ((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[2] = ((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];
		state[3] = ((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x];

		reduceDuplexRowSetupV2(state);

		uint32_t rowa;
		int prev = 3;

		for (int i = 0; i < 3; i++)
		{
			rowa = __shfl(state[0].x, 0, 4) & 3;
			reduceDuplexRowtV2(prev, rowa, i, state);
			prev = i;
		}

		rowa = __shfl(state[0].x, 0, 4) & 3;
		reduceDuplexRowtV2_4(rowa, state);

		((uint2*)DState)[(0 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[0];
		((uint2*)DState)[(1 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[1];
		((uint2*)DState)[(2 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[2];
		((uint2*)DState)[(3 * gridDim.x * blockDim.y + thread) * blockDim.x + threadIdx.x] = state[3];
	} //thread
}

__global__
__launch_bounds__(32, 1)
void lyra2v2_gpu_hash_32_3(uint32_t threads, uint32_t startNounce, uint2 *outputHash)
{
	const uint32_t thread = blockDim.x * blockIdx.x + threadIdx.x;

	uint28 state[4];

	if (thread < threads)
	{
		state[0] = __ldg4(&DState[blockDim.x * gridDim.x * 0 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[1] = __ldg4(&DState[blockDim.x * gridDim.x * 1 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[2] = __ldg4(&DState[blockDim.x * gridDim.x * 2 + blockDim.x * blockIdx.x + threadIdx.x]);
		state[3] = __ldg4(&DState[blockDim.x * gridDim.x * 3 + blockDim.x * blockIdx.x + threadIdx.x]);

		for (int i = 0; i < 12; i++)
			round_lyra_v35(state);

		outputHash[thread + threads * 0] = state[0].x;
		outputHash[thread + threads * 1] = state[0].y;
		outputHash[thread + threads * 2] = state[0].z;
		outputHash[thread + threads * 3] = state[0].w;

	} //thread
}

__host__
void lyra2v2_cpu_init(int thr_id, uint32_t threads, uint64_t *hash)
{
	cudaMemcpyToSymbol(DState, &hash, sizeof(hash), 0, cudaMemcpyHostToDevice);
}

__host__
void lyra2v2_cpu_hash_32(int thr_id, uint32_t threads, uint32_t startNounce, uint64_t *d_outputHash, uint32_t tpb)
{
	dim3 grid1((threads * 4 + tpb - 1) / tpb);
	dim3 block1(4, tpb >> 2);

	dim3 grid2((threads + tpb - 1) / tpb);
	dim3 block2(tpb);

	lyra2v2_gpu_hash_32_1 << <grid2, block2 >> > (threads, startNounce, (uint2*)d_outputHash);

	lyra2v2_gpu_hash_32_2 << <grid1, block1, 48 * sizeof(uint2) * tpb >> > (threads, startNounce, d_outputHash);

	lyra2v2_gpu_hash_32_3 << <grid2, block2 >> > (threads, startNounce, (uint2*)d_outputHash);

}

