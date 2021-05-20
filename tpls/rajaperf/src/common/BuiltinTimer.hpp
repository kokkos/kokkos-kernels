//
// Created by Poliakoff, David Zoeller on 4/26/21.
//
#include <chrono>
#include <limits>
#ifndef RAJAPERFSUITE_BUILTINTIMER_HPP
#define RAJAPERFSUITE_BUILTINTIMER_HPP
namespace rajaperf {
    class ChronoTimer {
    public:
        using ElapsedType = double;

    private:
        using ClockType = std::chrono::steady_clock;
        using TimeType = ClockType::time_point;
        using DurationType = std::chrono::duration<ElapsedType>;

    public:
        ChronoTimer() : tstart(ClockType::now()), tstop(ClockType::now()), telapsed(0) {
        }

        void start() { tstart = ClockType::now(); }

        void stop() {
            tstop = ClockType::now();
            telapsed +=
                    std::chrono::duration_cast<DurationType>(tstop - tstart).count();
        }

        ElapsedType elapsed() const { return telapsed; }

        void reset() { telapsed = 0; }

    private:
        TimeType tstart;
        TimeType tstop;
        ElapsedType telapsed;
    };
}
#endif //RAJAPERFSUITE_BUILTINTIMER_HPP
