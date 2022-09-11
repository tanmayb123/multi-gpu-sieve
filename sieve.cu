#include <atomic>
#include <chrono>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <sys/time.h>
#include <pthread.h>
#include <thread>

#define GPUS 2
#define BYTE unsigned char
#define THREADS 64
#define BLOCKS 256

inline BYTE *create_bitarray(size_t elements) {
    size_t bytes = elements / 8 + ((elements % 8) > 1);
    return (BYTE *) calloc(bytes, 1);
}

inline BYTE bitarray_get(BYTE *bitarray, size_t index) {
    size_t byte = index / 8;
    size_t offset_mask = 1 << (7 - (index % 8));
    return bitarray[byte] & offset_mask;
}

inline void bitarray_set(BYTE *bitarray, size_t index) {
    size_t byte = index / 8;
    size_t offset_mask = 1 << (7 - (index % 8));
    bitarray[byte] |= offset_mask;
}

void naive_sieve(uint64_t upper_bound, uint64_t **primes, uint64_t *prime_count) {
    BYTE *is_prime = create_bitarray(upper_bound);
    uint64_t total_primes = upper_bound - 2;
    for (uint64_t p = 2; (p * p) < upper_bound; p++) {
        if (bitarray_get(is_prime, p) == 0) {
            for (uint64_t i = p * p; i < upper_bound; i += p) {
                if (bitarray_get(is_prime, i) == 0) {
                    total_primes--;
                    bitarray_set(is_prime, i);
                }
            }
        }
    }

    *prime_count = total_primes;
    *primes = (uint64_t *) malloc(sizeof(uint64_t) * total_primes);
    size_t i = 0;
    for (uint64_t p = 2; p < upper_bound; p++) {
        if (bitarray_get(is_prime, p) == 0) (*primes)[i++] = p;
    }

    free(is_prime);
}

inline BYTE *create_bitarrays_gpu(size_t elements, size_t *bytes_per_bitarray, size_t bitarrays) {
    *bytes_per_bitarray = elements / 8 + ((elements % 8) > 1);
    BYTE *bitarrays_mem;
    cudaMalloc(&bitarrays_mem, *bytes_per_bitarray * bitarrays);
    return bitarrays_mem;
}

__global__ void sieve_chunk_gpu(BYTE *is_prime_arrays, size_t is_prime_bytes,
                                uint64_t default_prime_count, uint64_t *prime_counts,
                                uint64_t *seed_primes, uint64_t seed_count,
                                uint64_t chunk_size, uint64_t chunk_count, uint64_t chunk_offset) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= chunk_count) return;
    uint64_t offset_index = index + chunk_offset;

    BYTE *is_prime = is_prime_arrays + index * is_prime_bytes;
    
    uint64_t low = (offset_index + 1) * chunk_size;
    uint64_t high = low + chunk_size;
    low += 1 - (low & 1);
    high -= 1 - (high & 1);

    for (size_t i = 0; i < is_prime_bytes; i++) is_prime[i] = 0;

    uint64_t prime_count = default_prime_count;
    for (size_t i = 1; i < seed_count; i++) {
        uint64_t low_multiple = (uint64_t) (floor((double) low / (double) seed_primes[i]) * (double) seed_primes[i]);
        low_multiple += (low_multiple < low) * seed_primes[i];

        uint64_t j = low_multiple + (1 - (low_multiple & 1)) * seed_primes[i];
        while (j <= high) {
            uint64_t j_idx = (j - low) / 2;
            size_t byte = j_idx / 8;
            size_t offset_mask = 1 << (7 - (j_idx % 8));
            uint64_t is_set = (is_prime[byte] & offset_mask) == 0;
            prime_count -= is_set;
            BYTE potential_new_value = is_prime[byte] | offset_mask;
            is_prime[byte] = is_set * potential_new_value + (1 - is_set) * is_prime[byte];
            j += seed_primes[i] * 2;
        }
    }

    prime_counts[index] = prime_count;
}

uint64_t sieve_chunk_cpu(BYTE *is_prime, uint64_t *seed_primes, uint64_t seed_count, uint64_t chunk_size, uint64_t chunk_index) {
    uint64_t low = (chunk_index + 1) * chunk_size;
    uint64_t high = low + chunk_size;
    low += 1 - (low & 1);
    high -= 1 - (high & 1);

    for (size_t i = 1; i < seed_count; i++) {
        uint64_t low_multiple = (uint64_t) (floor((double) low / (double) seed_primes[i]) * (double) seed_primes[i]);
        low_multiple += (low_multiple < low) * seed_primes[i];

        uint64_t j = low_multiple + (1 - (low_multiple & 1)) * seed_primes[i];
        while (j <= high) {
            uint64_t j_idx = (j - low) / 2;
            size_t byte = j_idx / 8;
            size_t offset_mask = 1 << (7 - (j_idx % 8));
            uint64_t is_set = (is_prime[byte] & offset_mask) == 0;
            BYTE potential_new_value = is_prime[byte] | offset_mask;
            is_prime[byte] = is_set * potential_new_value + (1 - is_set) * is_prime[byte];
            j += seed_primes[i] * 2;
        }
    }

    return low;
}

typedef struct {
    int gpu;

    uint64_t chunk_size;
    uint64_t chunk_count;
    uint64_t chunk_offset;
    uint64_t chunk_prime_count;

    uint64_t *seed_primes;
    uint64_t seed_prime_count;

    uint64_t *chunk_prime_counts;

    std::atomic<uint64_t> *processed_chunks;
} gpu_worker_input;

void *process_chunks_on_gpu(void *vinput) {
    gpu_worker_input *input = (gpu_worker_input *) vinput;

    cudaSetDevice(input->gpu);

    uint64_t kernel_chunk_count = THREADS * BLOCKS;
    uint64_t invocations = 1;
    if (input->chunk_count < kernel_chunk_count) kernel_chunk_count = input->chunk_count;
    else invocations = input->chunk_count / kernel_chunk_count + ((input->chunk_count % kernel_chunk_count) > 0);

    uint64_t *seed_primes_gpu;
    size_t seed_primes_size = sizeof(uint64_t) * input->seed_prime_count;
    cudaMalloc(&seed_primes_gpu, seed_primes_size);
    cudaMemcpy(seed_primes_gpu, input->seed_primes, seed_primes_size, cudaMemcpyHostToDevice);

    size_t is_prime_bytes;
    BYTE *is_prime_arrays_gpu = create_bitarrays_gpu(input->chunk_prime_count, &is_prime_bytes, kernel_chunk_count);

    uint64_t *prime_counts_gpu;
    cudaMalloc(&prime_counts_gpu, sizeof(uint64_t) * kernel_chunk_count);

    uint64_t total_chunks_processed = 0;
    for (uint64_t i = 0; i < invocations; i++) {
        uint64_t offset = i * kernel_chunk_count;
        uint64_t remaining_chunks = input->chunk_count - offset;
        if (remaining_chunks > kernel_chunk_count) remaining_chunks = kernel_chunk_count;
        offset += input->chunk_offset;

        sieve_chunk_gpu<<<THREADS, BLOCKS>>>(
            is_prime_arrays_gpu, is_prime_bytes,
            input->chunk_prime_count, prime_counts_gpu,
            seed_primes_gpu, input->seed_prime_count,
            input->chunk_size, remaining_chunks, offset
        );
        cudaDeviceSynchronize();

        cudaMemcpy(input->chunk_prime_counts + total_chunks_processed, prime_counts_gpu, sizeof(uint64_t) * remaining_chunks, cudaMemcpyDeviceToHost);
        total_chunks_processed += remaining_chunks;
        *input->processed_chunks += remaining_chunks;
    }

    cudaFree(is_prime_arrays_gpu);
    cudaFree(seed_primes_gpu);

    return NULL;
}

void sieve(uint64_t m) {
    double dm = (double) m;
    uint64_t upper_bound = (uint64_t) (dm * log(dm)) + (dm * log(log(dm)));
    uint64_t chunk_size = (uint64_t) sqrt((double) upper_bound);
    uint64_t chunk_count = chunk_size - 1;
    uint64_t chunk_prime_count = chunk_size / 2 + chunk_size % 2;

    printf("Must process %lu chunks. Checking up to %lu.\n", chunk_count, upper_bound);

    uint64_t *seed_primes;
    uint64_t seed_prime_count;
    naive_sieve(chunk_size, &seed_primes, &seed_prime_count);

    uint64_t chunks_per_gpu = chunk_count / GPUS;
    uint64_t last_gpu_overflow = chunk_count % GPUS;
    uint64_t *chunk_prime_counts[GPUS];
    gpu_worker_input inputs[GPUS];
    pthread_t tids[GPUS];
    std::atomic<uint64_t> total_chunks_processed(0);

    for (size_t i = 0; i < GPUS; i++) {
        uint64_t gpu_chunk_count = chunks_per_gpu;
        if (i == (GPUS - 1)) gpu_chunk_count += last_gpu_overflow;

        chunk_prime_counts[i] = (uint64_t *) calloc(sizeof(uint64_t), gpu_chunk_count);

        inputs[i] = {
            (int) i,

            chunk_size,
            gpu_chunk_count,
            chunks_per_gpu * i,
            chunk_prime_count,

            seed_primes,
            seed_prime_count,

            chunk_prime_counts[i],

            &total_chunks_processed
        };

        pthread_create(tids + i, NULL, process_chunks_on_gpu, inputs + i);
    }

    while (total_chunks_processed < chunk_count) {
        printf("Processed: %lu\r", total_chunks_processed.load());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    printf("\n");

    for (size_t i = 0; i < GPUS; i++) pthread_join(tids[i], NULL);

    uint64_t total_chunks_checked = 0;
    uint64_t found_primes = seed_prime_count;
    for (size_t i = 0; i < GPUS; i++) {
        uint64_t gpu_chunk_count = chunks_per_gpu;
        if (i == (GPUS - 1)) gpu_chunk_count += last_gpu_overflow;

        for (size_t j = 0; j < gpu_chunk_count; j++) {
            uint64_t new_found = found_primes + chunk_prime_counts[i][j];
            if (new_found >= m) goto found_chunk;
            found_primes = new_found;
            total_chunks_checked++;
        }
    }

    BYTE *is_prime;
    uint64_t chunk_low;

    printf("Couldn't find nth prime.\n");
    goto cleanup;

found_chunk:
    is_prime = create_bitarray(chunk_prime_count);
    chunk_low = sieve_chunk_cpu(is_prime, seed_primes, seed_prime_count, chunk_size, total_chunks_checked);

    for (size_t i = 0; i < chunk_prime_count; i++) {
        if (bitarray_get(is_prime, i) == 0) found_primes++;
        if (found_primes == m) {
            printf("%lu\n", chunk_low + i * 2);
            break;
        }
    }

    free(is_prime);

cleanup:
    free(seed_primes);
}

int main() {
    struct timeval tv1, tv2;
    gettimeofday(&tv1, NULL);
    sieve(1e11);
    gettimeofday(&tv2, NULL);
    printf("Total time = %f seconds\n", (double) (tv2.tv_usec - tv1.tv_usec) / 1000000 + (double) (tv2.tv_sec - tv1.tv_sec));
}
