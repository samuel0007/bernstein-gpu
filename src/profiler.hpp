#pragma once

// ----------------------- activation -----------------------
#ifndef PROF_ENABLED
  #ifdef PROF_ACTIVATE
    #define PROF_ENABLED 1
  #else
    #define PROF_ENABLED 0
  #endif
#endif
// ----------------------------------------------------------

#if PROF_ENABLED  // ================= REAL IMPLEMENTATION =====================

#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <spdlog/spdlog.h>

#include "util.hpp"

class Prof {
public:
    static Prof& get() { static Prof p; return p; }

    // CPU
    int  cpu_begin(const std::string& tag, int lvl) {
        cpu_.push_back({tag, lvl, Clock::now(), {}});
        return (int)cpu_.size()-1;
    }
    void cpu_end(int id) { cpu_[id].t1 = Clock::now(); }

    void cpu_start(const std::string& tag, int lvl) {
        int id = cpu_begin(tag, lvl);
        cpu_open_[tag].push_back(id);
    }
    void cpu_stop(const std::string& tag) {
        auto& v = cpu_open_[tag];
        int id = v.back(); v.pop_back();
        cpu_end(id);
    }

    // GPU
    int  gpu_begin(const std::string& tag, int lvl, GpuStream s = 0) {
        GpuRange r; r.tag = tag; r.level = lvl; r.stream = s;
        GPU_EVENT_CREATE(r.e0); GPU_EVENT_CREATE(r.e1);
        GPU_EVENT_RECORD(r.e0, s);
        gpu_.push_back(r);
        return (int)gpu_.size()-1;
    }
    void gpu_end(int id) {
        auto& r = gpu_[id];
        GPU_EVENT_RECORD(r.e1, r.stream);
        stop_events_.push_back(r.e1);
    }
    void gpu_start(const std::string& tag, int lvl, GpuStream s = 0) {
        int id = gpu_begin(tag, lvl, s);
        gpu_open_[tag].push_back(id);
    }
    void gpu_stop(const std::string& tag) {
        auto& v = gpu_open_[tag];
        int id = v.back(); v.pop_back();
        gpu_end(id);
    }

    // finalize / report
    void finalize_gpu() {
        for (auto& e : stop_events_) GPU_EVENT_SYNC(e);
        for (auto& r : gpu_) GPU_EVENT_ELAPSE(r.ms, r.e0, r.e1);
    }

    void report()                          { report_range(0, std::numeric_limits<int>::max()); }
    void report_level(int lvl)             { report_range(lvl, lvl); }
    void report_below(int max_lvl)         { report_range(0, max_lvl); }
    void report_above(int min_lvl)         { report_range(min_lvl, std::numeric_limits<int>::max()); }
    void report_range(int min_lvl, int max_lvl) {
        finalize_gpu();
        for (auto& r : cpu_) {
            if (r.level < min_lvl || r.level > max_lvl) continue;
            if (r.t1.time_since_epoch().count()==0) continue;
            double ms = std::chrono::duration<double,std::milli>(r.t1 - r.t0).count();
            spdlog::info("[CPU][{}] {}: {} ms", r.level, r.tag, ms);
        }
        for (auto& r : gpu_) {
            if (r.level < min_lvl || r.level > max_lvl) continue;
            spdlog::info("[GPU][{}] {}: {} ms", r.level, r.tag, r.ms);
        }
    }

    // resets
    void reset_level(int lvl) {
        for (auto& r : gpu_) if (r.level == lvl) { GPU_EVENT_DESTROY(r.e0); GPU_EVENT_DESTROY(r.e1); }
        erase_level(cpu_, lvl);
        erase_level(gpu_, lvl);
        stop_events_.clear();
        cpu_open_.clear();
        gpu_open_.clear();
    }
    void reset_below(int keep_lvl) {
        for (auto& r : gpu_) if (r.level > keep_lvl) { GPU_EVENT_DESTROY(r.e0); GPU_EVENT_DESTROY(r.e1); }
        prune_cpu(keep_lvl);
        prune_gpu(keep_lvl);
    }
    void reset_all() { reset_below(-1); }

private:
    Prof() = default;
    Prof(const Prof&) = delete;
    Prof& operator=(const Prof&) = delete;

    using Clock = std::chrono::high_resolution_clock;

    struct CpuRange {
        std::string tag;
        int level;
        std::chrono::time_point<Clock> t0{}, t1{};
    };
    struct GpuRange {
        std::string tag;
        int level;
        GpuEvent e0{}, e1{};
        GpuStream stream{};
        float ms{0.f};
    };

    template<class Vec, class Tlevel>
    void erase_level(Vec& v, Tlevel lvl) {
        std::vector<typename Vec::value_type> kept;
        kept.reserve(v.size());
        for (auto& r : v) if (r.level != lvl) kept.push_back(r);
        v.swap(kept);
    }

    void prune_cpu(int keep_lvl) {
        std::vector<CpuRange> kept;
        kept.reserve(cpu_.size());
        for (auto& r : cpu_) if (r.level <= keep_lvl) kept.push_back(r);
        cpu_.swap(kept);
        cpu_open_.clear();
    }
    void prune_gpu(int keep_lvl) {
        std::vector<GpuRange> kept;
        kept.reserve(gpu_.size());
        for (auto& r : gpu_) if (r.level <= keep_lvl) kept.push_back(r);
        gpu_.swap(kept);
        stop_events_.clear();
        gpu_open_.clear();
    }

    std::vector<CpuRange> cpu_;
    std::vector<GpuRange> gpu_;
    std::vector<GpuEvent>  stop_events_;
    std::unordered_map<std::string, std::vector<int>> cpu_open_;
    std::unordered_map<std::string, std::vector<int>> gpu_open_;
};

// RAII + macros
namespace prof_detail {
    struct CpuGuard {
        int id{-1};
        CpuGuard(const std::string& tag, int lvl) { id = Prof::get().cpu_begin(tag, lvl); }
        ~CpuGuard() { Prof::get().cpu_end(id); }
    };
    struct GpuGuard {
        int id{-1};
        GpuGuard(const std::string& tag, int lvl, GpuStream s=0) { id = Prof::get().gpu_begin(tag, lvl, s); }
        ~GpuGuard() { Prof::get().gpu_end(id); }
    };
}

#define PROF_CAT2(a,b) a##b
#define PROF_CAT(a,b)  PROF_CAT2(a,b)

// RAII scopes
#define PROF_CPU_SCOPE(tag, lvl)             prof_detail::CpuGuard PROF_CAT(__cpu_scope_, __COUNTER__)(tag, lvl)
#define PROF_GPU_SCOPE(tag, lvl, stream)     prof_detail::GpuGuard PROF_CAT(__gpu_scope_, __COUNTER__)(tag, lvl, stream)

// Manual start/stop
#define PROF_CPU_START(tag, lvl)             Prof::get().cpu_start(tag, lvl)
#define PROF_CPU_STOP(tag)                   Prof::get().cpu_stop(tag)
#define PROF_GPU_START(tag, lvl, stream)     Prof::get().gpu_start(tag, lvl, stream)
#define PROF_GPU_STOP(tag)                   Prof::get().gpu_stop(tag)

// Report / finalize / reset
#define PROF_REPORT()                        Prof::get().report()
#define PROF_REPORT_LVL(lvl)               Prof::get().report_level(lvl)
#define PROF_REPORT_BELOW(maxlvl)            Prof::get().report_below(maxlvl)
#define PROF_REPORT_ABOVE(minlvl)            Prof::get().report_above(minlvl)
#define PROF_REPORT_RANGE(minlvl,maxlvl)     Prof::get().report_range(minlvl,maxlvl)
#define PROF_GPU_FINALIZE()                  Prof::get().finalize_gpu()
#define PROF_RESET_LVL(level)              Prof::get().reset_level(level)
#define PROF_RESET_BELOW(level)              Prof::get().reset_below(level)
#define PROF_RESET_ALL()                     Prof::get().reset_all()

#else   // ========================= STUB IMPLEMENTATION ========================

// Everything compiles away when PROF_ENABLED == 0

#define PROF_CPU_SCOPE(tag, lvl)             do{}while(0)
#define PROF_GPU_SCOPE(tag, lvl, stream)     do{}while(0)
#define PROF_CPU_START(tag, lvl)             do{}while(0)
#define PROF_CPU_STOP(tag)                   do{}while(0)
#define PROF_GPU_START(tag, lvl, stream)     do{}while(0)
#define PROF_GPU_STOP(tag)                   do{}while(0)
#define PROF_REPORT()                        do{}while(0)
#define PROF_REPORT_LVL(lvl)               do{}while(0)
#define PROF_REPORT_BELOW(maxlvl)            do{}while(0)
#define PROF_REPORT_ABOVE(minlvl)            do{}while(0)
#define PROF_REPORT_RANGE(minlvl,maxlvl)     do{}while(0)
#define PROF_GPU_FINALIZE()                  do{}while(0)
#define PROF_RESET_LVL(level)              do{}while(0)
#define PROF_RESET_BELOW(level)              do{}while(0)
#define PROF_RESET_ALL()                     do{}while(0)

#endif // PROF_ENABLED
