#include "mini_llama/backend.h"

#include <stdexcept>

#include "mini_llama/cuda_runtime.h"

namespace mini_llama {

// 只接受小写 cpu/cuda；失败时返回 false，并保留调用方原有的 kind
bool ParseBackendKind(const std::string& text, BackendKind& kind) {
  if (text == "cpu") {
    kind = BackendKind::kCpu;
    return true;
  }
  if (text == "cuda") {
    kind = BackendKind::kCuda;
    return true;
  }
  return false;
}

// 将后端枚举转换为 CLI 和追踪日志使用的名称，未知值交给校验函数拒绝
std::string BackendKindName(BackendKind kind) {
  switch (kind) {
    case BackendKind::kCpu:
      return "cpu";
    case BackendKind::kCuda:
      return "cuda";
  }
  return "unknown";
}

// 构建状态由 Runtime 模块决定，此处不查询驱动或 GPU
bool CudaBackendBuilt() { return CudaRuntimeBuilt(); }

// 按设备数量检查编号范围；未构建 CUDA 或查询失败时统一返回不可用
// 该检查不保证显存分配或推理一定成功，后续操作仍须检查各自的 API 错误
bool CudaBackendAvailable(int device_id) {
  try {
    const int count = CudaDeviceCount();
    return device_id >= 0 && device_id < count;
  } catch (const std::exception&) {
    return false;
  }
}

// 查询成功时格式化设备信息，失败时保留 Runtime 的具体错误原因供 CLI 展示
std::string CudaDeviceSummary(int device_id) {
  try {
    return CudaFormatDeviceInfo(CudaGetDeviceInfo(device_id));
  } catch (const std::exception& exception) {
    return exception.what();
  }
}

// 运行前校验配置与 CUDA 环境，不负责初始化推理上下文或上传模型权重
void ValidateBackend(const BackendConfig& config) {
  if (config.kind == BackendKind::kCpu) {
    // 依据用户是否显式传入参数判断冲突，显式 --device 0 也必须拒绝
    if (config.device_id_set) {
      throw std::runtime_error("--device can only be used with --backend cuda.");
    }
    return;
  }

  if (config.kind == BackendKind::kCuda) {
    // 库调用也可能传入非法编号，不能只依赖 CLI 的整数解析
    if (config.device_id < 0) {
      throw std::runtime_error("--device must be a non-negative integer GPU ID");
    }
    // 先检查构建配置，再检查设备；失败时抛异常，不回退到 CPU
    if (!CudaBackendBuilt()) {
      throw std::runtime_error(CudaDeviceSummary(config.device_id));
    }
    if (!CudaBackendAvailable(config.device_id)) {
      throw std::runtime_error(CudaDeviceSummary(config.device_id));
    }
    return;
  }

  // 枚举可能来自库调用，未知值必须拒绝，不能按 CUDA 或 CPU 处理
  throw std::runtime_error("unknown backend");
}

// 仅说明 CLI 执行路径，不查询环境
std::string BackendExecutionNote(const BackendConfig& config) {
  if (config.kind == BackendKind::kCpu) {
    return "compute: cpu";
  }
  if (config.kind == BackendKind::kCuda) {
    return "compute: cuda (device-resident decode; CPU sampler)";
  }
  throw std::runtime_error("unknown backend");
}

}  // namespace mini_llama
