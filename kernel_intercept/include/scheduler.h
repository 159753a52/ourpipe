/**
 * @file scheduler.h
 * @brief Orion GPU 调度器头文件（简化版）
 *
 * 架构：
 * - 单调度器线程轮询所有客户端队列
 * - HP 直接执行，BE 根据 Orion 逻辑判断
 * - 客户端提交后忙等完成
 */

#ifndef ORION_SCHEDULER_H
#define ORION_SCHEDULER_H

#include "gpu_capture.h"
#include "kernel_profile.h"
#include <cuda_runtime.h>
#include <thread>
#include <vector>
#include <atomic>
#include <string>
#include <fstream>

namespace orion {

// ============================================================================
// Kernel Profile 信息
// ============================================================================

/**
 * @brief Kernel 性能特征信息
 *
 * 从 kernel_info.csv 加载，用于 Orion 调度决策。
 */
struct KernelProfileInfo {
    std::string name;      // Kernel 名称
    int profile;           // 1: compute-bound, 0: mem-bound, -1: unclear
    int mem;               // 内存占用 (预留)
    int sm_used;           // 需要的 SM 数量
    float duration;        // 执行时间 (ns)

    KernelProfileInfo() : profile(-1), mem(0), sm_used(0), duration(0.0f) {}
    KernelProfileInfo(const std::string& n, int p, int m, int sm, float dur)
        : name(n), profile(p), mem(m), sm_used(sm), duration(dur) {}
};

// ============================================================================
// 调度配置（简化版）
// ============================================================================

struct SchedulerConfig {
    int device_id = 0;          // GPU 设备 ID
    int num_sms = 0;            // GPU SM 数量（运行时获取）
    int sm_threshold = 0;       // SM 阈值（固定值）
};

// ============================================================================
// Orion 调度状态（简化版）
// ============================================================================

struct OrionSchedulingState {
    // Kernel Profile 信息
    std::vector<std::vector<KernelProfileInfo>> op_info_vector;

    // 每个客户端当前已调度的 kernel 数
    std::vector<int> seen;

    // 每个客户端每次迭代的 kernel 数
    std::vector<int> num_client_kernels;

    // SM 阈值
    int sm_threshold = 0;

    // 保护状态的互斥锁
    std::mutex mutex;

    void init(int num_clients, int num_sms) {
        op_info_vector.resize(num_clients);
        seen.resize(num_clients, 0);
        num_client_kernels.resize(num_clients, 0);
        sm_threshold = num_sms / 2;  // 默认 50%
    }

    void reset() {
        for (auto& v : op_info_vector) v.clear();
        std::fill(seen.begin(), seen.end(), 0);
        std::fill(num_client_kernels.begin(), num_client_kernels.end(), 0);
    }
};

extern OrionSchedulingState g_orion_state;

// ============================================================================
// 调度器类（简化版）
// ============================================================================

class Scheduler {
public:
    Scheduler();
    ~Scheduler();

    bool init(int num_clients, const SchedulerConfig& config = SchedulerConfig());
    void start();
    void stop();
    void join();
    void reset();

    const SchedulerConfig& get_config() const { return config_; }
    SchedulerConfig& get_mutable_config() { return config_; }
    int get_num_clients() const { return num_clients_; }

    std::atomic<bool> running_{false};

private:
    void run();  // 单线程轮询（保留用于调试）
    void run_worker(int client_idx);  // 多 worker 模式：单个 worker 的主循环
    cudaError_t execute_operation(OperationPtr op, cudaStream_t stream);
    bool orion_should_schedule(OperationPtr op, int client_idx);
    bool create_streams();
    void destroy_streams();

public:
    // Streams (public for C interface access)
    cudaStream_t hp_stream_{nullptr};
    std::vector<cudaStream_t> be_streams_;

private:
    SchedulerConfig config_;
    std::thread thread_;  // 单线程模式使用
    std::vector<std::thread> workers_;  // 多 worker 模式使用
    std::atomic<bool> initialized_{false};

    int num_clients_{0};
};

extern Scheduler g_scheduler;

bool start_scheduler(int num_clients, const SchedulerConfig& config = SchedulerConfig());
void stop_scheduler();

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_start_scheduler(int num_clients, int device_id);
void orion_stop_scheduler();

// Kernel profile 加载
int orion_load_kernel_info(int client_idx, const char* file_path);

// 设置客户端 kernel 数
void orion_set_client_kernels(int client_idx, int num_kernels);

// 设置 SM 阈值
void orion_set_sm_threshold(int threshold);

// 获取 SM 阈值
int orion_get_sm_threshold();

// 同步特定客户端的 stream（只等待该客户端的操作完成）
void orion_sync_client_stream(int client_idx);

// 重置状态
void orion_reset_state();

}

#endif // ORION_SCHEDULER_H
