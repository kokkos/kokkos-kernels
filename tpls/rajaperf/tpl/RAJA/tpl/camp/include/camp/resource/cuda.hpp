/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_CUDA_HPP
#define __CAMP_CUDA_HPP

#include "camp/defines.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_CUDA
#include <cuda_runtime.h>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    namespace
    {
      struct device_guard {
        device_guard(int device)
        {
          cudaGetDevice(&prev_device);
          cudaSetDevice(device);
        }

        ~device_guard() { cudaSetDevice(prev_device); }

      int prev_device;
    };

    }  // namespace

    class CudaEvent
    {
    public:
      CudaEvent(cudaStream_t stream)
      {
        cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming);
        cudaEventRecord(m_event, stream);
      }
      bool check() const { return (cudaEventQuery(m_event) == cudaSuccess); }
      void wait() const { cudaEventSynchronize(m_event); }
      cudaEvent_t getCudaEvent_t() const { return m_event; }

    private:
      cudaEvent_t m_event;
    };

    class Cuda
    {
      static cudaStream_t get_a_stream(int num)
      {
        static cudaStream_t streams[16] = {};
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        std::call_once(m_onceFlag, [] {
          if (streams[0] == nullptr) {
            for (auto &s : streams) {
              cudaStreamCreate(&s);
            }
          }
        });

        if (num < 0) {
          m_mtx.lock();
          previous = (previous + 1) % 16;
          m_mtx.unlock();
          return streams[previous];
        }

        return streams[num % 16];
      }

    private:
      Cuda(cudaStream_t s, int dev=0) : stream(s), device(dev) {}
    public:
      Cuda(int group = -1, int dev=0) : stream(get_a_stream(group)), device(dev) {}

      // Methods
      Platform get_platform() { return Platform::cuda; }
      static Cuda &get_default()
      {
        static Cuda c( [] {
          cudaStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          cudaStreamCreate(&s);
#endif
          return s;
        }());
        return c;
      }

      CudaEvent get_event()
      {
        auto d{device_guard(device)};
        return CudaEvent(get_stream());
      }

      Event get_event_erased()
      {
        auto d{device_guard(device)};
        return Event{CudaEvent(get_stream())};
      }

      void wait()
      {
        auto d{device_guard(device)};
        cudaStreamSynchronize(stream);
      }

      void wait_for(Event *e)
      {
        auto *cuda_event = e->try_get<CudaEvent>();
        if (cuda_event) {
          auto d{device_guard(device)};
          cudaStreamWaitEvent(get_stream(), cuda_event->getCudaEvent_t(), 0);
        } else {
          e->wait();
        }
      }

      // Memory
      template <typename T>
      T *allocate(size_t size)
      {
        T *ret = nullptr;
        if (size > 0) {
          auto d{device_guard(device)};
          cudaMallocManaged(&ret, sizeof(T) * size);
        }
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p)
      {
        auto d{device_guard(device)};
        cudaFree(p);
      }
      void memcpy(void *dst, const void *src, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          cudaMemcpyAsync(dst, src, size, cudaMemcpyDefault, stream);
        }
      }
      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          auto d{device_guard(device)};
          cudaMemsetAsync(p, val, size, stream);
        }
      }

      cudaStream_t get_stream() { return stream; }
      int get_device() { return device; }

    private:
      cudaStream_t stream;
      int device;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_CUDA

#endif /* __CAMP_CUDA_HPP */
