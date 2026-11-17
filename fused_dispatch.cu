#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned long long u64;
#define THREADS_PER_BLOCK 256 //Should be variable maybe? 
#define DEBUG_FLAG true
#define MAX_RANKS 8
#define WAIT_MAX_SPINS 50000000LL //timeout so a bug shows up as a error not a hang


struct Meta { int src_row; int expert; float weight; };

struct DeviceContext {
    //input on this this GPU
    float *tokens;                      // [num_tokens * token_Length]
    int *topk_idx;                      // [num_tokens * k]
    float *topk_w;                      // [num_tokens * k]
    //recieve side, on this GPU: one lane per source GPU
    float *recv_payload[MAX_RANKS];     // each [capacity * token_length]
    Meta * recv_meta[MAX_RANKS];        // each [capacity]
    //send side, on this GPU
    float **payload_table;              // [n_ranks] device array of pointers
    Meta **meta_table;                  // [n_ranks]
    int *send_count;                    // [n_ranks]
    u64 *recv_hdr;                      // [n_ranks] on this gpu, 1 header word per source lane.        Mailbox flags that gets written to by each publish kernel (corresponding rank to corresponding flag)
    int *recv_count;                    // [n_ranks] filled by the wait kernel                          
    int *wait_error;                    // [1] non-zero if a wait timed out                             
    u64 **hdr_table;                    // [n_ranks] device array, hdr_table[dst] = my header word on dst           Mapping to mailbox flags to write to.

};

static DeviceContext g_ctx[MAX_RANKS];
static int g_ranks, g_num_tokens, g_K, g_token_size, g_experts_per_rank, g_capacity;
static int g_initialized = 0;

__global__ void scatter_forward_kernel(
    const float* tokens,        // [num_tokens, token_length]
    const int* topk_idx,        // [num_tokens, k]
    const float* topk_w,        // [num_tokens, k]
    float* const* payload_lane, // [nranks] #Pointer to address on each gpu where this kernel should write its tokens to.
    Meta* const* meta_lane,     // [nranks] #Pointer to address on each gpu where this kernel should write its meta-info about the tokens to.
    int* send_count,            // [nranks], local to this gpu. Use for atomic adds when sending, the returned value ensures its unique address
    int num_tokens, int K, int token_length, int experts_per_rank)

{
        /*kernel SCATTER   (one block per (token t, choice k) pair)
        e    = topk_idx[t][k]
        dst  = e / experts_per_rank
        thread 0 only:
            slot = atomicAdd(&send_count[dst], 1)      // local atomic
            meta_lane[dst][slot] = {t, e, topk_w[t][k]}
            put slot in shared memory
        __syncthreads()
        all threads, strided over H:
            payload_lane[dst][slot][i] = h[t][i]
    */
    int token_length_idx = threadIdx.x;
    int pair = blockIdx.x; //(token * choice), same as token_t * k + choice_k
    int token_t = pair / K;
    //int choice_k = pair % K;

    int expert_choice = topk_idx[pair];
    int dest_rank = expert_choice / experts_per_rank;

    __shared__ int slot;

    if(token_length_idx == 0) {
        int destination_slot = atomicAdd(&send_count[dest_rank], 1); //local atomic, increments destination/sent counter, returning the number at that time, which gives us a unique slot for the token
        meta_lane[dest_rank][destination_slot] = {token_t, expert_choice, topk_w[pair]}; //send meta info about what token is in that slot.
        
        slot = destination_slot; //Put slot in shared memory for all threads
    }
    __syncthreads();

    for(int i = threadIdx.x; i < token_length; i += blockDim.x) {
            payload_lane[dest_rank][slot * token_length + i] = tokens[token_t * token_length + i];
    }

}

//one thread per destination. Runs after scatter on the same stream, so send_count is final when this reads it.
__global__ void publish_kernel(u64 *const *hdr_table, int *send_count, int n_ranks, unsigned int seq) 
{
    int destination = threadIdx.x;
    
    if(destination >= n_ranks) {
        return;
    } 

    u64 word = ((u64)seq << 32) | (unsigned int)send_count[destination]; //Seq is in upper bits, doesn't interfere with count
    *hdr_table[destination] = word; //single 8-byte store in dst's memory
    send_count[destination] = 0; //ready for the next step 
}

// one thread per source. spins until that surce has published this step
__global__ void wait_kernel(const u64 *recv_hdr, int *recv_count, int *wait_error, int n_ranks, unsigned int seq, long long max_spins) 
{
    int source = threadIdx.x;
    if(source >= n_ranks) {
        return;
    }

    const volatile u64 *hdr = recv_hdr + source;   //volatile: re-read memory every iteration
    for(long long spin = 0; spin < max_spins; spin++) {
        u64 word = *hdr;
        if((unsigned int)(word >> 32) == seq) {
            recv_count[source] = (int)(word & 0xffffffffULL);
            return;
        }
    }
    recv_count[source] = -1;    //timed out
    *wait_error = 1;
}


static void check_cuda(cudaError_t err, const char *where)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(err));
        exit(1);
    }
}

static void *dev_alloc(size_t bytes, const char *what) {
    void *p = NULL;
    check_cuda(cudaMalloc(&p, bytes), what);
    return p;
}

// Call after every kernel launch: reports launch errors, does not wait 
static void check_launch(const char *where)
{
    check_cuda(cudaGetLastError(), where); //catches bad launch configs, does not wait
}



// C-style wrapper functions for ctypes. extern "C" keeps the names unmangled
// so ctypes can find them.
extern "C"
{
    void dispatch_free();

    void set_inputs(int device, const float *tokens, const int *topk_idx, const float *topk_w) {
        check_cuda(cudaSetDevice(device), "set_inputs");
        DeviceContext &c = g_ctx[device];
        size_t pairs = (size_t)g_num_tokens * g_K;

        check_cuda(cudaMemcpy(c.tokens, tokens, (size_t)g_num_tokens * g_token_size * sizeof(float),
                                cudaMemcpyHostToDevice), "set_inputs: tokens");
        check_cuda(cudaMemcpy(c.topk_idx, topk_idx, pairs * sizeof(int),
                                cudaMemcpyHostToDevice), "set_inputs: topk_idx");
        check_cuda(cudaMemcpy(c.topk_w, topk_w, pairs * sizeof(float), 
                                cudaMemcpyHostToDevice), "set_inputs: topk_w");
    }
    //reads the lane of 'device' that was written by 'src'
    void read_lane(int device, int src, int count, float *payload_out, Meta *meta_out) {
        if (count < 0 || count > g_capacity) {
            fprintf(stderr, "read_lane: count %d exceeds capacity %d\n", count, g_capacity);
            exit(1);
        }
        check_cuda(cudaSetDevice(device), "read_lane");
        DeviceContext &c = g_ctx[device];
        
        check_cuda(cudaMemcpy(payload_out, c.recv_payload[src], (size_t)count * g_token_size * sizeof(float),
                            cudaMemcpyDeviceToHost), "read_lane: payload");
        check_cuda(cudaMemcpy(meta_out, c.recv_meta[src], (size_t)count * sizeof(Meta),
                            cudaMemcpyDeviceToHost), "read_lane: meta");
    }

    void read_send_counts(int device, int *out)
    {
        check_cuda(cudaSetDevice(device), "read_send_counts");
        check_cuda(cudaMemcpy(out, g_ctx[device].send_count, g_ranks * sizeof(int),
                            cudaMemcpyDeviceToHost), "read_send_counts");
    }

    void reset_counts(int device)
    {
        check_cuda(cudaSetDevice(device), "reset_counts");
        check_cuda(cudaMemset(g_ctx[device].send_count, 0, g_ranks * sizeof(int)), "reset_counts");
    }

    void dispatch_init(int n_ranks, int num_tokens, int K, int token_size, int num_experts) {
        if (n_ranks > MAX_RANKS || num_experts % n_ranks != 0) {
            fprintf(stderr, "dispatch_init: bad n_ranks or num_experts\n");
            exit(1);
        }
        if(g_initialized) {
            dispatch_free();
        }
        g_ranks = n_ranks;
        g_num_tokens = num_tokens;
        g_K = K;
        g_token_size = token_size;
        g_experts_per_rank = num_experts / n_ranks;
        g_capacity = num_tokens * K; //Worst case: every pair goes to one GPU

        // Peer access, both directions for every pair.
        for(int d = 0; d < n_ranks; d++) {
            check_cuda(cudaSetDevice(d), "init: set_device");
            for (int p = 0; p < n_ranks; p++) {
                if(p == d) continue;
                cudaError_t err = cudaDeviceEnablePeerAccess(p,0);
                if(err == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();
                }
                if (err != cudaErrorPeerAccessAlreadyEnabled) {
                    check_cuda(err, "init: enable peer access");
                }
            }
        }

        

        //Allocate everyhing that lives on each device
        size_t pairs = (size_t)num_tokens * K;
        for(int d = 0; d < n_ranks; d++) {
            check_cuda(cudaSetDevice(d), "init set device");
            DeviceContext &c = g_ctx[d];
            c.tokens = (float *)dev_alloc((size_t)num_tokens*token_size*sizeof(float), "tokens");
            c.topk_idx = (int *)dev_alloc(pairs * sizeof(int), "topk_idx");
            c.topk_w = (float *)dev_alloc(pairs * sizeof(float), "topk_w");

            for (int src = 0; src < n_ranks; src++) {
                c.recv_payload[src] = (float *)dev_alloc((size_t)g_capacity * token_size * sizeof(float), "recv_payload");
                c.recv_meta[src] = (Meta *) dev_alloc((size_t)g_capacity * sizeof(Meta), "recv_meta");
            }

            c.send_count = (int *)dev_alloc(n_ranks * sizeof(int), "send_count");
            check_cuda(cudaMemset(c.send_count, 0, n_ranks * sizeof(int)), "zero send_count");

            c.payload_table = (float **)dev_alloc(n_ranks * sizeof(float *), "payload_table");
            c.meta_table = (Meta **)dev_alloc(n_ranks * sizeof(Meta *), "meta_table");

            c.recv_hdr = (u64 *)dev_alloc(n_ranks * sizeof(u64), "recv_hdr");
            check_cuda(cudaMemset(c.recv_hdr, 0, n_ranks * sizeof(u64)), "zero recv_hdr");
            c.recv_count = (int *)dev_alloc(n_ranks * sizeof(int), "recv_count");
            c.wait_error = (int *)dev_alloc(sizeof(int), "wait_error");
            check_cuda(cudaMemset(c.wait_error, 0, sizeof(int)), "zero wait_error");
            c.hdr_table = (u64 **)dev_alloc(n_ranks * sizeof(u64 *), "hdr_table");
        }

        //Fill each sender's tables with the address of its lanes on every destination
        for(int me = 0; me < n_ranks; me++) {
            float *tmp_payload[MAX_RANKS];
            Meta *tmp_meta[MAX_RANKS];
            u64 *tmp_hdr[MAX_RANKS];
            for(int dst = 0; dst < n_ranks; dst++) {
                tmp_payload[dst] = g_ctx[dst].recv_payload[me];
                tmp_meta[dst] = g_ctx[dst].recv_meta[me];
                tmp_hdr[dst] = g_ctx[dst].recv_hdr + me; //address of 1 word, my slot on dst
            }
            check_cuda(cudaSetDevice(me), "init: set device");
            check_cuda(cudaMemcpy(g_ctx[me].payload_table, tmp_payload,
                                    n_ranks * sizeof(float *), cudaMemcpyHostToDevice), "fill payload table");
            check_cuda(cudaMemcpy(g_ctx[me].meta_table, tmp_meta,
                                    n_ranks * sizeof(Meta *), cudaMemcpyHostToDevice), "fill meta_table");
            check_cuda(cudaMemcpy(g_ctx[me].hdr_table, tmp_hdr,
                                    n_ranks * sizeof(u64 *), cudaMemcpyHostToDevice), "fill hdr_table");
            
        }
        g_initialized = 1;
    }

    void dispatch_free() {
        if(!g_initialized) {
            return;
        }
        
        for (int d = 0; d < g_ranks; d++) {
            check_cuda(cudaSetDevice(d), "dispatch_free");
            check_cuda(cudaDeviceSynchronize(), "dispatch_free: sync");
        }
        
        for (int d = 0; d < g_ranks; d++) {
            check_cuda(cudaSetDevice(d), "dispatch_free");
            DeviceContext &c = g_ctx[d];
            cudaFree(c.tokens); cudaFree(c.topk_idx); cudaFree(c.topk_w);
            for(int src = 0; src < g_ranks; src++) {
                cudaFree(c.recv_payload[src]);
                cudaFree(c.recv_meta[src]);
            }
            cudaFree(c.payload_table); cudaFree(c.meta_table); cudaFree(c.hdr_table);
            cudaFree(c.send_count); cudaFree(c.recv_hdr);
            cudaFree(c.recv_count); cudaFree(c.wait_error);
            memset(&c, 0, sizeof(c));
        }
        g_initialized = 0;
    }


    void dispatch_scatter(int device) {
        check_cuda(cudaSetDevice(device), "dispatch_scatter");
        DeviceContext &c = g_ctx[device];

        dim3 blocks(g_num_tokens * g_K);    //one block per (token, choice) pair
        dim3 threads(THREADS_PER_BLOCK);    

        scatter_forward_kernel<<<blocks, threads>>>(
            c.tokens, c.topk_idx, c.topk_w,
            c.payload_table, c.meta_table, c.send_count,
            g_num_tokens, g_K, g_token_size, g_experts_per_rank);
        
        check_launch("dispatch_scatter");
        if (DEBUG_FLAG) check_cuda(cudaDeviceSynchronize(), "dispatch_scatter: sync");
    }

    void dispatch_publish(int device, unsigned int seq) {
        check_cuda(cudaSetDevice(device), "dispatch_publish");
        DeviceContext &c = g_ctx[device];
        publish_kernel<<<1, g_ranks>>>(c.hdr_table, c.send_count, g_ranks, seq);
        check_launch("dispatch_publish");
    }

    // Never syncs. This kernel waits on other devices, so blocking here can deadlock.
    void dispatch_wait(int device, unsigned int seq) {
        check_cuda(cudaSetDevice(device), "dispatch_wait");
        DeviceContext &c = g_ctx[device];
        wait_kernel<<<1, g_ranks>>>(c.recv_hdr, c.recv_count, c.wait_error, g_ranks, seq, WAIT_MAX_SPINS);
        check_launch("dispatch_wait");
    }

    void dispatch_sync_all() {
        for (int d = 0; d < g_ranks; d++) {
            check_cuda(cudaSetDevice(d), "dispatch_sync_all");
            check_cuda(cudaDeviceSynchronize(), "dispatch_sync_all");
        }
    }

    // returns the wait_error flag; fills out[src] with the count received from each source 
    int read_recv_counts(int device, int *out) {
        int err = 0;
        check_cuda(cudaSetDevice(device), "read_recv_counts");
        check_cuda(cudaMemcpy(out, g_ctx[device].recv_count, g_ranks * sizeof(int),
                                cudaMemcpyDeviceToHost), "read_recv_counts");
        check_cuda(cudaMemcpy(&err, g_ctx[device].wait_error, sizeof(int),
                                cudaMemcpyDeviceToHost), "read_recv_counts: error flag");
        return err;
    }



    // Utility functions

    int get_cuda_device_count()
    {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess)
        {
            return 0;
        }
        return count;
    }
}