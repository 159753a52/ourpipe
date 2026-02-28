/**
 * @file scheduler.cpp
 * @brief Orion GPU 调度器实现（简化版）
 *
 * 架构：
 * - 单调度器线程轮询所有客户端队列
 * - HP (client 0) 直接执行
 * - BE (client 1+) 根据 Orion 逻辑判断：SM 阈值 + Profile 互补
 * - 内存操作直接执行
 * - 没有 HP kernel 在执行时，BE 可以直接执行
 */

#include "scheduler.h"
#include <algorithm>
#include <cstring>

// cuDNN/cuBLAS 状态类型（避免引入头文件依赖）
typedef int cudnnStatus_t;
typedef int cublasStatus_t;

namespace orion {

// ============================================================================
// 全局变量
// ============================================================================

Scheduler g_scheduler;
OrionSchedulingState g_orion_state;

// ============================================================================
// 外部函数声明
// ============================================================================

extern cudaError_t execute_cuda_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cudnnStatus_t execute_cudnn_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cublasStatus_t execute_cublas_operation(OperationPtr op, cudaStream_t scheduler_stream);
extern cublasStatus_t execute_cublaslt_operation(OperationPtr op, cudaStream_t scheduler_stream);

// 线程局部变量
using orion::tl_is_scheduler_thread;

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * @brief 检查是否是内存操作（直接执行，不走调度判断）
 */
static bool is_memory_operation(OperationType type) {
    return type == OperationType::MALLOC ||
           type == OperationType::FREE ||
           type == OperationType::MEMCPY ||
           type == OperationType::MEMCPY_ASYNC ||
           type == OperationType::MEMSET ||
           type == OperationType::MEMSET_ASYNC;
}

/**
 * @brief 检查是否是需要调度的 kernel 操作
 */
static bool is_kernel_operation(OperationType type) {
    switch (type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
        case OperationType::CUBLASLT_MATMUL:
            return true;
        default:
            return false;
    }
}

/**
 * @brief 从 kernel_info.csv 加载 kernel profile 信息
 */
static int populate_kernel_info(int client_idx, const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open kernel_info file: %s", file_path.c_str());
        return -1;
    }

    std::vector<KernelProfileInfo> op_info;
    std::string line;
    bool first_line = true;

    while (std::getline(file, line)) {
        if (first_line) {
            first_line = false;
            continue;
        }
        if (line.empty()) continue;

        // 解析 CSV: Name,Profile,Memory_footprint,SM_usage,Duration
        // 从右边解析最后 4 个字段（Name 可能包含逗号）
        std::vector<size_t> comma_positions;
        for (size_t i = 0; i < line.size(); i++) {
            if (line[i] == ',') {
                comma_positions.push_back(i);
            }
        }

        if (comma_positions.size() < 4) {
            LOG_WARN("Invalid CSV line: %s", line.c_str());
            continue;
        }

        size_t n = comma_positions.size();
        size_t pos_profile = comma_positions[n - 4];
        size_t pos_mem = comma_positions[n - 3];
        size_t pos_sm = comma_positions[n - 2];
        size_t pos_dur = comma_positions[n - 1];

        std::string name = line.substr(0, pos_profile);
        std::string profile_str = line.substr(pos_profile + 1, pos_mem - pos_profile - 1);
        std::string mem_str = line.substr(pos_mem + 1, pos_sm - pos_mem - 1);
        std::string sm_str = line.substr(pos_sm + 1, pos_dur - pos_sm - 1);
        std::string dur_str = line.substr(pos_dur + 1);

        KernelProfileInfo info;
        info.name = name;
        try {
            info.profile = std::stoi(profile_str);
            info.mem = std::stoi(mem_str);
            info.sm_used = std::stoi(sm_str);
            info.duration = std::stof(dur_str);
        } catch (const std::exception& e) {
            LOG_WARN("Failed to parse CSV line: %s", line.c_str());
            continue;
        }

        op_info.push_back(info);
    }

    file.close();

    {
        std::lock_guard<std::mutex> lock(g_orion_state.mutex);
        g_orion_state.op_info_vector[client_idx] = std::move(op_info);
    }

    size_t count = g_orion_state.op_info_vector[client_idx].size();
    LOG_INFO("Loaded %zu kernel profiles for client %d from %s", count, client_idx, file_path.c_str());
    return (int)count;
}

// ============================================================================
// Scheduler 实现
// ============================================================================

Scheduler::Scheduler() {}

Scheduler::~Scheduler() {
    stop();
    join();
    destroy_streams();
}

bool Scheduler::init(int num_clients, const SchedulerConfig& config) {
    if (initialized_.load()) {
        LOG_WARN("Scheduler already initialized");
        return true;
    }

    num_clients_ = num_clients;
    config_ = config;

    // 获取 GPU SM 数量
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to get device properties: %s", cudaGetErrorString(err));
        return false;
    }
    config_.num_sms = prop.multiProcessorCount;
    LOG_INFO("GPU has %d SMs", config_.num_sms);

    // 设置默认 SM 阈值
    if (config_.sm_threshold == 0) {
        config_.sm_threshold = config_.num_sms / 2;
    }

    if (!create_streams()) {
        return false;
    }

    // 初始化 Orion 状态
    g_orion_state.init(num_clients, config_.num_sms);
    g_orion_state.sm_threshold = config_.sm_threshold;

    initialized_.store(true);
    LOG_INFO("Scheduler initialized: %d clients, SM threshold=%d", num_clients, config_.sm_threshold);
    return true;
}

bool Scheduler::create_streams() {
    // 【修复】在创建流之前设置设备
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to set device %d for stream creation: %s",
                  config_.device_id, cudaGetErrorString(err));
        return false;
    }

    int lowest_priority, highest_priority;
    cudaDeviceGetStreamPriorityRange(&lowest_priority, &highest_priority);
    // CUDA: 数值越小优先级越高
    // highest_priority 通常是 -1，lowest_priority 通常是 0

    LOG_INFO("Stream priority range: [%d (highest), %d (lowest)]", highest_priority, lowest_priority);

    // HP stream (最高优先级)
    err = cudaStreamCreateWithPriority(&hp_stream_, cudaStreamNonBlocking, highest_priority);
    if (err != cudaSuccess) {
        LOG_ERROR("Failed to create HP stream: %s", cudaGetErrorString(err));
        return false;
    }
    LOG_INFO("HP stream created with priority %d", highest_priority);

    // BE streams (优先级递减：BE1 > BE2 > BE3 ...)
    // 在 (highest_priority, lowest_priority] 范围内分配
    int num_be = num_clients_ - 1;
    be_streams_.resize(num_be);

    int priority_range = lowest_priority - highest_priority;  // 可用优先级级别数

    for (int i = 0; i < num_be; i++) {
        // BE 优先级从 highest_priority + 1 开始，逐渐递减到 lowest_priority
        // BE1 优先级最高（数值最小），BEn 优先级最低（数值最大）
        int be_priority;
        if (priority_range > 0 && num_be > 0) {
            // 在范围内均匀分配，确保不超过 lowest_priority
            be_priority = highest_priority + 1 + (i * (priority_range - 1)) / std::max(1, num_be - 1);
            be_priority = std::min(be_priority, lowest_priority);
        } else {
            // 如果没有优先级范围，所有 BE 使用 lowest_priority
            be_priority = lowest_priority;
        }

        err = cudaStreamCreateWithPriority(&be_streams_[i], cudaStreamNonBlocking, be_priority);
        if (err != cudaSuccess) {
            LOG_ERROR("Failed to create BE stream %d: %s", i, cudaGetErrorString(err));
            return false;
        }
        LOG_INFO("BE%d stream created with priority %d", i + 1, be_priority);
    }

    LOG_DEBUG("Created streams: 1 HP + %d BE", (int)be_streams_.size());
    return true;
}

void Scheduler::destroy_streams() {
    if (hp_stream_) {
        cudaStreamDestroy(hp_stream_);
        hp_stream_ = nullptr;
    }
    for (auto& stream : be_streams_) {
        if (stream) {
            cudaStreamDestroy(stream);
        }
    }
    be_streams_.clear();
}

void Scheduler::start() {
    if (!initialized_.load()) {
        LOG_ERROR("Scheduler not initialized");
        return;
    }
    if (running_.load()) {
        LOG_WARN("Scheduler already running");
        return;
    }

    running_.store(true);
    
    // 【多 worker 模式】启动 num_clients_ 个 worker 线程（每个 micro-batch 一个）
    workers_.resize(num_clients_);
    for (int i = 0; i < num_clients_; i++) {
        workers_[i] = std::thread(&Scheduler::run_worker, this, i);
    }
    LOG_INFO("Scheduler started with %d worker threads (multi-priority stream mode)", num_clients_);
}

void Scheduler::stop() {
    if (!running_.load()) return;

    LOG_INFO("Stopping scheduler...");
    running_.store(false);

    // 关闭所有队列（唤醒可能在等待的 worker）
    for (int i = 0; i < num_clients_; i++) {
        if (g_capture_state.client_queues[i]) {
            g_capture_state.client_queues[i]->shutdown();
        }
    }
}

void Scheduler::join() {
    // 【多 worker 模式】等待所有 worker 线程结束
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    
    // 兼容单线程模式
    if (thread_.joinable()) {
        thread_.join();
    }
    LOG_INFO("All scheduler workers joined");
}

void Scheduler::reset() {
    if (running_.load()) {
        LOG_WARN("Cannot reset while running");
        return;
    }
    destroy_streams();
    initialized_.store(false);
    num_clients_ = 0;
    g_orion_state.reset();
    LOG_INFO("Scheduler reset");
}

/**
 * @brief Orion 调度判断：BE kernel 是否可以执行
 *
 * 简化版判断条件：
 * 1. 内存操作：直接允许
 * 2. 非 kernel 操作：直接允许
 * 3. HP 队列为空：允许 BE 执行
 * 4. HP 队列不为空：BE 等待
 */
bool Scheduler::orion_should_schedule(OperationPtr op, int client_idx) {
    (void)client_idx;  // 暂时不使用

    // 内存操作直接允许
    if (is_memory_operation(op->type)) {
        return true;
    }

    // 非 kernel 操作直接允许（如同步操作）
    if (!is_kernel_operation(op->type)) {
        return true;
    }

    // 检查 HP 队列是否为空
    // 如果 HP 队列为空，说明 HP 当前没有待执行的操作，BE 可以执行
    if (g_capture_state.client_queues[0] &&
        g_capture_state.client_queues[0]->empty()) {
        return true;
    }

    // HP 队列不为空，BE 需要等待
    return false;
}

/**
 * @brief 执行操作
 */
cudaError_t Scheduler::execute_operation(OperationPtr op, cudaStream_t stream) {
    op->started.store(true);

    switch (op->type) {
        case OperationType::KERNEL_LAUNCH:
        case OperationType::MALLOC:
        case OperationType::FREE:
        case OperationType::MEMCPY:
        case OperationType::MEMCPY_ASYNC:
        case OperationType::MEMSET:
        case OperationType::MEMSET_ASYNC:
        case OperationType::DEVICE_SYNC:
        case OperationType::STREAM_SYNC:
            return execute_cuda_operation(op, stream);

        case OperationType::CUDNN_CONV_FWD:
        case OperationType::CUDNN_CONV_BWD_DATA:
        case OperationType::CUDNN_CONV_BWD_FILTER:
        case OperationType::CUDNN_BATCHNORM_FWD:
        case OperationType::CUDNN_BATCHNORM_BWD:
            return execute_cudnn_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLAS_SGEMM:
        case OperationType::CUBLAS_SGEMM_BATCHED:
        case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED:
            return execute_cublas_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        case OperationType::CUBLASLT_MATMUL:
            return execute_cublaslt_operation(op, stream) == 0 ? cudaSuccess : cudaErrorUnknown;

        default:
            LOG_ERROR("Unknown operation type: %d", (int)op->type);
            return cudaErrorUnknown;
    }
}

// 外部声明：预初始化 cuBLAS handle
extern void preinit_thread_local_cublas_handle();

/**
 * @brief 单线程轮询调度器主循环（保留用于调试和回退）
 */
void Scheduler::run() {
    tl_is_scheduler_thread = true;
    LOG_INFO("Scheduler thread running (single-threaded polling mode - legacy)");

    // 【修复】不要在调度器线程中显式设置设备
    // 原因：cudaSetDevice() 会影响全局 CUDA 设备状态，可能破坏主线程的 CUDA 上下文
    // 解决方案：流已经在 create_streams() 时与正确的设备绑定，不需要再次设置
    // 只在必要时初始化 CUDA 上下文（但不改变设备）
    
    // 使用 cudaDeviceSynchronize() 初始化 CUDA 上下文（但不改变设备）
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_WARN("Scheduler thread: cudaDeviceSynchronize returned %d: %s",
                  err, cudaGetErrorString(err));
    }

    // 预初始化 cuBLAS handle（避免首次调用时的竞争条件）
    preinit_thread_local_cublas_handle();
    LOG_INFO("Scheduler thread: cuBLAS handle pre-initialized");

    while (running_.load()) {
        bool did_work = false;

        // 轮询所有客户端队列
        for (int i = 0; i < num_clients_; i++) {
            if (!g_capture_state.client_queues[i]) continue;

            // peek 查看队首操作
            auto op = g_capture_state.client_queues[i]->peek();
            if (!op) continue;

            // 选择 stream
            cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];

            bool should_execute = false;

            if (i == 0) {
                // HP: 直接执行
                should_execute = true;
            } else {
                // BE: 直接执行（与 HP 相同，不做任何处理）
                // 原 BE 调度逻辑已注释：
                should_execute = orion_should_schedule(op, i);
                // should_execute = true;
            }

            if (should_execute) {
                // 从队列移除
                g_capture_state.client_queues[i]->try_pop();

                // 【方案B】根据操作类型选择 stream：优先使用原始 stream
                cudaStream_t exec_stream = stream;
                switch (op->type) {
                    case OperationType::KERNEL_LAUNCH: {
                        auto& p = std::get<KernelLaunchParams>(op->params);
                        if (p.stream) exec_stream = p.stream;
                        break;
                    }
                    case OperationType::CUBLAS_SGEMM:
                    case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                        auto& p = std::get<CublasGemmParams>(op->params);
                        if (p.original_stream) exec_stream = p.original_stream;
                        break;
                    }
                    case OperationType::CUBLASLT_MATMUL: {
                        auto& p = std::get<CublasLtMatmulParams>(op->params);
                        if (p.stream) exec_stream = p.stream;
                        break;
                    }
                    default:
                        break;
                }

                // 执行操作
                cudaError_t err = execute_operation(op, exec_stream);

                // 更新 seen 计数
                if (is_kernel_operation(op->type)) {
                    std::lock_guard<std::mutex> lock(g_orion_state.mutex);
                    g_orion_state.seen[i]++;
                }

                // 标记完成（客户端忙等检测此标志）
                op->mark_completed(err);
                did_work = true;
            }
        }

        // 如果没有工作，短暂让出 CPU
        if (!did_work) {
            // 纯忙等，不 sleep，最小化延迟
            std::this_thread::yield();
        }
    }

    // 处理剩余操作
    for (int i = 0; i < num_clients_; i++) {
        if (!g_capture_state.client_queues[i]) continue;

        cudaStream_t stream = (i == 0) ? hp_stream_ : be_streams_[i - 1];
        while (!g_capture_state.client_queues[i]->empty()) {
            auto op = g_capture_state.client_queues[i]->try_pop();
            if (op) {
                // 【方案B】同样使用原始 stream
                cudaStream_t exec_stream = stream;
                switch (op->type) {
                    case OperationType::KERNEL_LAUNCH: {
                        auto& p = std::get<KernelLaunchParams>(op->params);
                        if (p.stream) exec_stream = p.stream;
                        break;
                    }
                    case OperationType::CUBLAS_SGEMM:
                    case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                        auto& p = std::get<CublasGemmParams>(op->params);
                        if (p.original_stream) exec_stream = p.original_stream;
                        break;
                    }
                    case OperationType::CUBLASLT_MATMUL: {
                        auto& p = std::get<CublasLtMatmulParams>(op->params);
                        if (p.stream) exec_stream = p.stream;
                        break;
                    }
                    default:
                        break;
                }
                cudaError_t err = execute_operation(op, exec_stream);
                op->mark_completed(err);
            }
        }
        cudaStreamSynchronize(stream);
    }

    LOG_INFO("Scheduler thread exiting");
}

/**
 * @brief 多 worker 模式：单个 worker 线程的主循环
 *
 * 每个 worker 负责处理一个 client 队列的操作。
 * 不做 HP/BE 调度判断，优先级完全靠 CUDA stream priority 控制。
 *
 * @param client_idx 该 worker 负责的 client 索引
 */
void Scheduler::run_worker(int client_idx) {
    tl_is_scheduler_thread = true;
    tl_worker_idx = client_idx;  // 设置当前 worker 索引
    LOG_INFO("Worker %d started (stream priority mode)", client_idx);

    // 线程级 CUDA 初始化（只做一次）
    cudaError_t err = cudaSetDevice(config_.device_id);
    if (err != cudaSuccess) {
        LOG_ERROR("Worker %d: cudaSetDevice(%d) failed: %s",
                  client_idx, config_.device_id, cudaGetErrorString(err));
        return;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        LOG_WARN("Worker %d: cudaDeviceSynchronize returned %d: %s",
                  client_idx, err, cudaGetErrorString(err));
    }

    // 预初始化 cuBLAS handle（避免首次调用时的竞争条件）
    preinit_thread_local_cublas_handle();
    LOG_INFO("Worker %d: cuBLAS handle pre-initialized on device %d", client_idx, config_.device_id);

    // 获取该 worker 对应的队列和 stream
    auto* q = g_capture_state.client_queues[client_idx].get();
    if (!q) {
        LOG_ERROR("Worker %d: queue is null", client_idx);
        return;
    }
    
    cudaStream_t stream = (client_idx == 0) ? hp_stream_ : be_streams_[client_idx - 1];
    LOG_INFO("Worker %d: using stream %p (client_idx=%d)", client_idx, stream, client_idx);

    while (running_.load()) {
        auto op = q->try_pop();
        if (!op) {
            std::this_thread::yield();
            continue;
        }

        // 【方案B】根据操作类型选择 stream：优先使用原始 stream（Python 的 stream）
        cudaStream_t exec_stream = stream;  // 默认使用 Orion 的 stream
        
        switch (op->type) {
            case OperationType::KERNEL_LAUNCH: {
                auto& p = std::get<KernelLaunchParams>(op->params);
                if (p.stream) {
                    exec_stream = p.stream;
                }
                break;
            }
            case OperationType::CUBLAS_SGEMM:
            case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                auto& p = std::get<CublasGemmParams>(op->params);
                if (p.original_stream) {
                    exec_stream = p.original_stream;
                }
                break;
            }
            case OperationType::CUBLASLT_MATMUL: {
                auto& p = std::get<CublasLtMatmulParams>(op->params);
                if (p.stream) {
                    exec_stream = p.stream;
                }
                break;
            }
            case OperationType::MEMCPY_ASYNC:
            case OperationType::MEMSET_ASYNC: {
                auto& p = std::get<MemcpyParams>(op->params);
                if (p.stream) {
                    exec_stream = p.stream;
                }
                break;
            }
            default:
                // 其他操作使用默认的 Orion stream
                break;
        }

        // 执行操作（使用原始 stream 或 Orion stream）
        cudaError_t exec_err = execute_operation(op, exec_stream);

        // 更新 seen 计数
        if (is_kernel_operation(op->type)) {
            std::lock_guard<std::mutex> lock(g_orion_state.mutex);
            g_orion_state.seen[client_idx]++;
        }

        // 标记完成（客户端忙等检测此标志）
        op->mark_completed(exec_err);
    }

    // 处理剩余操作
    LOG_INFO("Worker %d: processing remaining operations", client_idx);
    while (!q->empty()) {
        auto op = q->try_pop();
        if (op) {
            // 【方案B】同样使用原始 stream
            cudaStream_t exec_stream = stream;
            switch (op->type) {
                case OperationType::KERNEL_LAUNCH: {
                    auto& p = std::get<KernelLaunchParams>(op->params);
                    if (p.stream) exec_stream = p.stream;
                    break;
                }
                case OperationType::CUBLAS_SGEMM:
                case OperationType::CUBLAS_SGEMM_STRIDED_BATCHED: {
                    auto& p = std::get<CublasGemmParams>(op->params);
                    if (p.original_stream) exec_stream = p.original_stream;
                    break;
                }
                case OperationType::CUBLASLT_MATMUL: {
                    auto& p = std::get<CublasLtMatmulParams>(op->params);
                    if (p.stream) exec_stream = p.stream;
                    break;
                }
                default:
                    break;
            }
            cudaError_t exec_err = execute_operation(op, exec_stream);
            op->mark_completed(exec_err);
        }
    }
    // 同步所有可能使用的 stream
    cudaStreamSynchronize(stream);

    LOG_INFO("Worker %d exiting", client_idx);
}

// ============================================================================
// 便捷函数
// ============================================================================

bool start_scheduler(int num_clients, int device_id) {
    SchedulerConfig config;
    config.device_id = device_id;
    
    if (init_capture_layer(num_clients) != 0) {
        LOG_ERROR("Failed to initialize capture layer");
        return false;
    }

    if (!g_scheduler.init(num_clients, config)) {
        LOG_ERROR("Failed to initialize scheduler");
        return false;
    }

    g_scheduler.start();
    return true;
}

void stop_scheduler() {
    g_scheduler.stop();
    g_scheduler.join();
    g_scheduler.reset();
    shutdown_capture_layer();
}

} // namespace orion

// ============================================================================
// C 接口
// ============================================================================

extern "C" {

int orion_start_scheduler(int num_clients, int device_id) {
    return orion::start_scheduler(num_clients, device_id) ? 0 : -1;
}

void orion_stop_scheduler() {
    orion::stop_scheduler();
}

int orion_load_kernel_info(int client_idx, const char* file_path) {
    if (client_idx < 0 || client_idx >= (int)orion::g_orion_state.op_info_vector.size()) {
        LOG_ERROR("Invalid client_idx %d", client_idx);
        return -1;
    }
    return orion::populate_kernel_info(client_idx, std::string(file_path));
}

void orion_set_client_kernels(int client_idx, int num_kernels) {
    if (client_idx < 0 || client_idx >= (int)orion::g_orion_state.num_client_kernels.size()) {
        LOG_ERROR("Invalid client_idx %d", client_idx);
        return;
    }
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    orion::g_orion_state.num_client_kernels[client_idx] = num_kernels;
    LOG_INFO("Client %d: num_kernels=%d", client_idx, num_kernels);
}

void orion_set_sm_threshold(int threshold) {
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    orion::g_orion_state.sm_threshold = threshold;
    orion::g_scheduler.get_mutable_config().sm_threshold = threshold;
    LOG_INFO("SM threshold set to %d", threshold);
}

int orion_get_sm_threshold() {
    std::lock_guard<std::mutex> lock(orion::g_orion_state.mutex);
    return orion::g_orion_state.sm_threshold;
}

void orion_sync_client_stream(int client_idx) {
    // 同步特定客户端的 stream
    // client_idx == 0: HP stream
    // client_idx > 0: BE stream
    if (client_idx == 0) {
        if (orion::g_scheduler.hp_stream_) {
            cudaStreamSynchronize(orion::g_scheduler.hp_stream_);
        }
    } else {
        int be_idx = client_idx - 1;
        if (be_idx < (int)orion::g_scheduler.be_streams_.size() &&
            orion::g_scheduler.be_streams_[be_idx]) {
            cudaStreamSynchronize(orion::g_scheduler.be_streams_[be_idx]);
        }
    }
}

void orion_reset_state() {
    orion::g_orion_state.reset();
    LOG_INFO("Orion state reset");
}

} // extern "C"
