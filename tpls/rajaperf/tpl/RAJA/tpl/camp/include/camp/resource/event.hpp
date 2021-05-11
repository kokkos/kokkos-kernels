/*
Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory
Maintained by Tom Scogland <scogland1@llnl.gov>
CODE-756261, All rights reserved.
This file is part of camp.
For details about use and distribution, please read LICENSE and NOTICE from
http://github.com/llnl/camp
*/

#ifndef __CAMP_EVENT_HPP
#define __CAMP_EVENT_HPP

namespace camp
{
namespace resources
{
  inline namespace v1
  {

    class Event
    {
    public:
      Event() {}
      template <typename T>
      explicit Event(T &&value)
      {
        m_value.reset(new EventModel<T>(value));
      }

      bool check() const { return m_value->check(); }
      void wait() const { m_value->wait(); }

      template <typename T>
      T *try_get()
      {
        auto result = dynamic_cast<EventModel<T> *>(m_value.get());
        return result->get();
      }
      template <typename T>
      T get()
      {
        auto result = dynamic_cast<EventModel<T> *>(m_value.get());
        if (result == nullptr) {
          throw std::runtime_error("Incompatible Event type get cast.");
        }
        return *result->get();
      }

    private:
      class EventInterface
      {
      public:
        virtual ~EventInterface() {}
        virtual bool check() const = 0;
        virtual void wait() const = 0;
      };

      template <typename T>
      class EventModel : public EventInterface
      {
      public:
        EventModel(T const &modelVal) : m_modelVal(modelVal) {}
        bool check() const override { return m_modelVal.check(); }
        void wait() const override { m_modelVal.wait(); }
        T *get() { return &m_modelVal; }

      private:
        T m_modelVal;
      };

      std::shared_ptr<EventInterface> m_value;
    };

  }  // namespace v1
}  // namespace resources
}  // namespace camp
#endif /* __CAMP_EVENT_HPP */
