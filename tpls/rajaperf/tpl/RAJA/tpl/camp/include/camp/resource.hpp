/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_RESOURCE_HPP
#define __CAMP_RESOURCE_HPP

#include <cstring>
#include <memory>
#include <mutex>
#include <type_traits>

#include "camp/helpers.hpp"
#include "camp/resource/event.hpp"
#include "camp/resource/host.hpp"

#if defined(CAMP_HAVE_CUDA)
#include "camp/resource/cuda.hpp"
#endif
#if defined(CAMP_HAVE_HIP)
#include "camp/resource/hip.hpp"
#endif
#if defined(CAMP_HAVE_OMP_OFFLOAD)
#include "camp/resource/omp_target.hpp"
#endif

// last to ensure we don't hide breakage in the others
#include "camp/resource/platform.hpp"

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class Resource
    {
    public:
      Resource(Resource &&) = default;
      Resource(Resource const &) = default;
      Resource &operator=(Resource &&) = default;
      Resource &operator=(Resource const &) = default;
      template <typename T,
                typename = typename std::enable_if<
                    !std::is_same<typename std::decay<T>::type,
                                  Resource>::value>::type>
      Resource(T &&value)
      {
        m_value.reset(new ContextModel<type::ref::rem<T>>(forward<T>(value)));
      }
      template <typename T>
      T *try_get()
      {
        auto result = dynamic_cast<ContextModel<T> *>(m_value.get());
        return result ? result->get() : nullptr;
      }
      template <typename T>
      T get()
      {
        auto result = dynamic_cast<ContextModel<T> *>(m_value.get());
        if (result == nullptr) {
          throw std::runtime_error("Incompatible Resource type get cast.");
        }
        return *result->get();
      }
      Platform get_platform() { return m_value->get_platform(); }
      template <typename T>
      T *allocate(size_t size)
      {
        return (T *)m_value->calloc(size * sizeof(T));
      }
      void *calloc(size_t size) { return m_value->calloc(size); }
      void deallocate(void *p) { m_value->deallocate(p); }
      void memcpy(void *dst, const void *src, size_t size)
      {
        m_value->memcpy(dst, src, size);
      }
      void memset(void *p, int val, size_t size)
      {
        m_value->memset(p, val, size);
      }
      Event get_event() { return m_value->get_event(); }
      void wait_for(Event *e) { m_value->wait_for(e); }

    private:
      class ContextInterface
      {
      public:
        virtual ~ContextInterface() {}
        virtual Platform get_platform() = 0;
        virtual void *calloc(size_t size) = 0;
        virtual void deallocate(void *p) = 0;
        virtual void memcpy(void *dst, const void *src, size_t size) = 0;
        virtual void memset(void *p, int val, size_t size) = 0;
        virtual Event get_event() = 0;
        virtual void wait_for(Event *e) = 0;
      };

      template <typename T>
      class ContextModel : public ContextInterface
      {
      public:
        ContextModel(T const &modelVal) : m_modelVal(modelVal) {}
        Platform get_platform() override { return m_modelVal.get_platform(); }
        void *calloc(size_t size) override { return m_modelVal.calloc(size); }
        void deallocate(void *p) override { m_modelVal.deallocate(p); }
        void memcpy(void *dst, const void *src, size_t size) override
        {
          m_modelVal.memcpy(dst, src, size);
        }
        void memset(void *p, int val, size_t size) override
        {
          m_modelVal.memset(p, val, size);
        }
        Event get_event() override { return m_modelVal.get_event_erased(); }
        void wait_for(Event *e) override { m_modelVal.wait_for(e); }
        T *get() { return &m_modelVal; }

      private:
        T m_modelVal;
      };

      std::shared_ptr<ContextInterface> m_value;
    };

    template <Platform p>
    struct resource_from_platform;
    template <>
    struct resource_from_platform<Platform::host> {
      using type = ::camp::resources::Host;
    };
#if defined(CAMP_HAVE_CUDA)
    template <>
    struct resource_from_platform<Platform::cuda> {
      using type = ::camp::resources::Cuda;
    };
#endif
#if defined(CAMP_HAVE_HIP)
    template <>
    struct resource_from_platform<Platform::hip> {
      using type = ::camp::resources::Hip;
    };
#endif
#if defined(CAMP_HAVE_OMP_OFFLOAD)
    template <>
    struct resource_from_platform<Platform::omp_target> {
      using type = ::camp::resources::Omp;
    };
#endif

  template<typename Res>
  struct EventProxy {
    EventProxy(EventProxy &&) = default;
    EventProxy(EventProxy const &) = delete;
    EventProxy &operator=(EventProxy &&) = default;
    EventProxy &operator=(EventProxy const &) = delete;

    EventProxy(Res* r) :
      resource_{r}
    {}

    Event get() const {
      return resource_->get_event_erased();
    }

    operator Event() const {
      return resource_->get_event_erased();
    }

    Res* resource_;
  };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_RESOURCE_HPP */
