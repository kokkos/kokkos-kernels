/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_HIP_HPP
#define __CAMP_HIP_HPP

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

#ifdef CAMP_HAVE_HIP
#include <hip/hip_runtime.h>

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class HipEvent
    {
    public:
      HipEvent(hipStream_t stream)
      {
        hipEventCreateWithFlags(&m_event, hipEventDisableTiming);
        hipEventRecord(m_event, stream);
      }
      bool check() const { return (hipEventQuery(m_event) == hipSuccess); }
      void wait() const { hipEventSynchronize(m_event); }
      hipEvent_t getHipEvent_t() const { return m_event; }

    private:
      hipEvent_t m_event;
    };

    class Hip
    {
      static hipStream_t get_a_stream(int num)
      {
        static hipStream_t streams[16] = {};
        static int previous = 0;

        static std::once_flag m_onceFlag;
        static std::mutex m_mtx;

        std::call_once(m_onceFlag, [] {
          if (streams[0] == nullptr) {
            for (auto &s : streams) {
              hipStreamCreate(&s);
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
      Hip(hipStream_t s) : stream(s) {}
    public:
      Hip(int group = -1) : stream(get_a_stream(group)) {}

      // Methods
      Platform get_platform() { return Platform::hip; }
      static Hip &get_default()
      {
        static Hip h( [] {
          hipStream_t s;
#if CAMP_USE_PLATFORM_DEFAULT_STREAM
          s = 0;
#else
          hipStreamCreate(&s);
#endif
          return s;
        }());
        return h;
      }
      HipEvent get_event() { return HipEvent(get_stream()); }
      Event get_event_erased() { return Event{HipEvent(get_stream())}; }
      void wait() { hipStreamSynchronize(stream); }
      void wait_for(Event *e)
      {
        auto *hip_event = e->try_get<HipEvent>();
        if (hip_event) {
          hipStreamWaitEvent(get_stream(),
                              hip_event->getHipEvent_t(),
                              0);
        } else {
          e->wait();
        }
      }

      // Memory
      template <typename T>
      T *allocate(size_t size)
      {
        T *ret = nullptr;
        hipMallocManaged(&ret, sizeof(T) * size);
        return ret;
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p) { hipFree(p); }
      void memcpy(void *dst, const void *src, size_t size)
      {
        hipMemcpyAsync(dst, src, size, hipMemcpyDefault, stream);
      }
      void memset(void *p, int val, size_t size)
      {
        hipMemsetAsync(p, val, size, stream);
      }

      hipStream_t get_stream() { return stream; }

    private:
      hipStream_t stream;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif  //#ifdef CAMP_HAVE_HIP

#endif /* __CAMP_HIP_HPP */
