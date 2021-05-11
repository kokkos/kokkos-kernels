/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_HOST_HPP
#define __CAMP_HOST_HPP

#include "camp/resource/event.hpp"
#include "camp/resource/platform.hpp"

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class HostEvent
    {
    public:
      HostEvent() {}
      bool check() const { return true; }
      void wait() const {}
    };

    class Host
    {
    public:
      Host(int /* group */ = -1) {}

      // Methods
      Platform get_platform() { return Platform::host; }
      static Host &get_default()
      {
        static Host h;
        return h;
      }
      HostEvent get_event() { return HostEvent(); }
      Event get_event_erased()
      {
        Event e{HostEvent()};
        return e;
      }
      void wait() {}
      void wait_for(Event *e) { e->wait(); }

      // Memory
      template <typename T>
      T *allocate(size_t n)
      {
        return (T *)malloc(sizeof(T) * n);
      }
      void *calloc(size_t size)
      {
        void *p = allocate<char>(size);
        this->memset(p, 0, size);
        return p;
      }
      void deallocate(void *p) { free(p); }
      void memcpy(void *dst, const void *src, size_t size) { std::memcpy(dst, src, size); }
      void memset(void *p, int val, size_t size) { std::memset(p, val, size); }
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_DEVICES_HPP */
