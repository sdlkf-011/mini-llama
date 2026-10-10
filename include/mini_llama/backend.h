#ifndef MINI_LLAMA_INCLUDE_MINI_LLAMA_BACKEND_H_
#define MINI_LLAMA_INCLUDE_MINI_LLAMA_BACKEND_H_

#include <string>

namespace mini_llama {

enum class BackendKind {
  kCpu,   // CPU 后端，默认使用
  kCuda,  // CUDA 后端，需具备对应实现和可用 GPU
};

struct BackendConfig {
  BackendKind kind = BackendKind::kCpu;  // 未传入 --backend 时使用 CPU
  int device_id = 0;                     // CUDA 设备编号，默认选择设备 0
  bool device_id_set = false;            // 用户显式传入 --device 时为 true
};

// 解析小写 cpu/cuda；成功时更新 kind，失败时返回 false 并保留原值
bool ParseBackendKind(const std::string& text, BackendKind& kind);

// 返回后端名称 cpu/cuda；未知枚举值返回 unknown
std::string BackendKindName(BackendKind kind);

// 判断是否启用 CUDA 构建；不代表 CLI CUDA 推理已经接入
bool CudaBackendBuilt();

// 判断指定 CUDA 设备是否可用，默认检查设备 0
bool CudaBackendAvailable(int device_id = 0);

// 返回指定 CUDA 设备的摘要，或后端、设备不可用的原因
std::string CudaDeviceSummary(int device_id = 0);

// 校验后端配置和运行环境；CPU 禁止显式指定设备，无法运行时抛出异常
void ValidateBackend(const BackendConfig& config);

// 说明 CLI 计算路径；CUDA 推理未接入时返回不可用说明，未知后端抛出异常
// 此接口不校验环境，运行前应单独调用 ValidateBackend
std::string BackendExecutionNote(const BackendConfig& config);

}  // namespace mini_llama

#endif  // MINI_LLAMA_INCLUDE_MINI_LLAMA_BACKEND_H_
