#include <cstdlib>
#include <immintrin.h>
#include <thread>
#include <vector>
#include <random>
#include <iostream>
#include <functional>
#include <unistd.h>
#include <mutex>
#include <memory.h>

#include "Task.h"

using namespace std;


const uint16_t C_simd_width = sizeof(__m256) / sizeof(float);

static __m256i simd_masks[] = {
    _mm256_setr_epi32(-1,  0,  0,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1,  0,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1,  0,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1,  0,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1,  0,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1,  0, 0),
    _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0)
};

std::vector<std::thread> threads;
vector<float> times;
std::mutex times_mutex;


inline uint64_t getPerformanceCounter() {
  uint32_t low, high;
  asm volatile ("rdtsc" : "=a" (low), "=d" (high));
  return (((uint64_t) high) << 32) | low;
}

std::vector<int> bounds(int parts, int mem) {
    std::vector<int>bnd;
    int delta = mem / parts;
    int reminder = mem % parts;
    int N1 = 0, N2 = 0;
    bnd.push_back(N1);
    for (int i = 0; i < parts; ++i) {
        N2 = N1 + delta;
        if (i == parts - 1)
            N2 += reminder;
        bnd.push_back(N2);
        N1 = N2;
    }
    return bnd;
}

template <class T> void fill_data_buffer(T array[], int array_size, T range_min, T range_max) {
  static mt19937 seed(1);
  std::uniform_real_distribution<T> dist(range_min, range_max);
  
  for(int i=0; i<array_size; ++i) {
    array[i] = static_cast<T>(dist(seed));
  }
}

void calculate_activation_result(uint16_t vector_size, float *input)
{
  if(vector_size == 0) return;

  __m256 zero = _mm256_setzero_ps();

  if(vector_size == C_simd_width) {
      __m256 dst = _mm256_max_ps(zero, _mm256_loadu_ps(input));
      _mm256_storeu_ps(input, dst);
  }
  else {
      __m256 dst = _mm256_max_ps(zero, _mm256_maskload_ps(input, simd_masks[vector_size-1]));
      _mm256_maskstore_ps(input, simd_masks[vector_size-1], dst);
  }
}

void worker(float* array, int idx, int begin, int end) {

  volatile uint64_t pre = getPerformanceCounter();
  
  //std::cout << "Thread #" << idx << ": on CPU " << sched_getcpu() << "\n";

/*
  __m256 zero = _mm256_setzero_ps();
  int size = (end - begin)/C_simd_width;
  for(int i=0; i<size; ++i) {
    __m256 dst = _mm256_max_ps(zero, _mm256_loadu_ps(array));
    _mm256_storeu_ps(array, dst);
    array += C_simd_width;
  }
*/
  volatile uint64_t post = getPerformanceCounter();
  times.push_back(post - pre);
  
  //for(int i=0; i<10000000000; ++i) ;

/*
  cpu_set_t      l_cpuSet;

  CPU_ZERO(&l_cpuSet);
  printf("get affinity %d\n", pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &l_cpuSet));
  // printf("cpuset %d\n",l_cpuSet);
  printf("thread id %u\n", pthread_self());

  if ( pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &l_cpuSet ) == 0 )
      for (int i = 0; i < 4; i++)
          if (CPU_ISSET(i, &l_cpuSet))
              printf("XXX CPU: CPU %d\n", i);
*/
}

void process_chunk_of_task_data(const Task &task) {
  
  //volatile uint64_t pre = getPerformanceCounter();

  //std::thread::id this_id = std::this_thread::get_id();

  cpu_set_t cpu;
  CPU_ZERO(&cpu);
  CPU_SET(task.thread_idx, &cpu);
  int temp = pthread_setaffinity_np(threads[task.thread_idx].native_handle(), sizeof(cpu_set_t), &cpu);
  if(temp != 0) printf("ERROR setaffinity\n");
  
  timespec ts_beg, ts_end;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_beg);

  auto input = task.buffer + task.offset;
  auto chunks_count = task.size/C_simd_width;
  
  //printf("%d\n", chunks_count);

  for (auto i = 0u; i < chunks_count; ++i) {
      calculate_activation_result(C_simd_width, input);
      input += C_simd_width;
  }
  calculate_activation_result(task.size % C_simd_width, input);

  //volatile uint64_t post = getPerformanceCounter();
  //times.push_back(post - pre);
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts_end);
  //std::cout << (ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9 << " sec" << '\n';
  times_mutex.lock();
  times.push_back((ts_end.tv_sec - ts_beg.tv_sec) + (ts_end.tv_nsec - ts_beg.tv_nsec) / 1e9);
  times_mutex.unlock();
}

int main(int argc, char** argv) {

  int num_threads = 1;
  unsigned long buffer_size = 1000000000;

  if(argc > 1) buffer_size = strtoul(argv[1], NULL, 0);
  if(argc > 2) num_threads = atoi(argv[2]);

  printf("num_threads: %d, buffer_size: %lu\n", num_threads, buffer_size);

  buffer_size *= C_simd_width;

  //Init data buffer
  unique_ptr<float[]> buffer(new float[buffer_size]);
  //fill_data_buffer<float>(buffer.get(), buffer_size, -10, 10);
  memset(buffer.get(), 0, buffer_size*sizeof(float));

  //unsigned num_cpus = std::thread::hardware_concurrency();
  //std::cout << "CPUS: " << num_cpus << '\n';

  auto chunk_size = buffer_size/num_threads;

  //Create tasks
  std::vector<Task> tasks;

  tasks.resize(num_threads);
  for (auto i = 0; i < tasks.size(); ++i) {
    auto offset = i * chunk_size;
    auto size = (i < num_threads-1) ? (i+1) * chunk_size - offset : buffer_size - offset;

    tasks[i] = {i, buffer.get(), size, offset};
  }

  //Create threads
  threads.resize(tasks.size());
  for(int i=0; i<tasks.size(); ++i) {
    threads[i] = std::thread(process_chunk_of_task_data, tasks[i]);
  }

  for(auto &th : threads) th.join();

  //for(auto time : times) printf("%lf\n", time);
  printf("Avg: %lf\n", std::accumulate(times.begin(), times.end(), (float)0)/num_threads);

  //for (int j = 0; j < CPU_SETSIZE; j++)
  //    if (CPU_ISSET(j, &cpu))
  //        printf("Affinity CPU %d\n", j);

  return 0;
}
