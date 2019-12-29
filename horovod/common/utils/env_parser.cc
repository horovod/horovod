// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ============================================================================

#include "env_parser.h"

#include <cstring>
#include <iostream>
#include <stdlib.h>

#include "../logging.h"
#include "../operations.h"
#include "../stall_inspector.h"

namespace horovod {
namespace common {

std::string TypeName(LibType type) {
  switch (type) {
  case LibType::MPI:
    return std::string(HOROVOD_MPI);
  case LibType::GLOO:
    return std::string(HOROVOD_GLOO);
  case LibType::CCL:
    return std::string(HOROVOD_CCL);
  default:
    return std::string("Unknown");
  }
}

LibType ParseCPUOpsFromEnv() {
  // set default cpu operations for data transferring
  LibType cpu_operation = LibType::MPI;
#if HAVE_CCL
  cpu_operation = LibType::CCL;
#endif

  // If specified by admin during compiling
#if HOROVOD_CPU_OPERATIONS_DEFAULT == 'M'
  cpu_operation = LibType::MPI;
#elif HOROVOD_CPU_OPERATIONS_DEFAULT == 'G'
  cpu_operation = LibType::GLOO;
#elif HOROVOD_CPU_OPERATIONS_DEFAULT == 'C'
  cpu_operation = LibType::CCL;
#endif

  // If specified by user during runtime
  const char* user_cpu_operation = std::getenv(HOROVOD_CPU_OPERATIONS);
  if (user_cpu_operation != nullptr) {
    if (strcasecmp(user_cpu_operation, HOROVOD_MPI) == 0) {
      cpu_operation = LibType::MPI;
    } else if (strcasecmp(user_cpu_operation, HOROVOD_GLOO) == 0) {
      cpu_operation = LibType::GLOO;
    } else if (strcasecmp(user_cpu_operation, HOROVOD_CCL) == 0) {
      cpu_operation = LibType::CCL;
    } else {
      throw std::runtime_error("Unsupported CPU operation type, only MPI, "
                               "oneCCL, and Gloo are supported");
    }
  }

  LOG(DEBUG) << "Using " << TypeName(cpu_operation)
            << " to perform CPU operations.";
  return cpu_operation;
}

LibType ParseControllerOpsFromEnv() {
  // Always default to MPI if available.
  LibType controller;
#if HAVE_MPI
  controller = LibType::MPI;
#elif HAVE_GLOO
  controller = LibType::GLOO;
#endif

  // If specified during compilation
#if HOROVOD_CONTROLLER_DEFAULT == 'G'
  controller = LibType::GLOO;
#elif HOROVOD_CONTROLLER_DEFAULT == 'M'
  controller = LibType::MPI;
#endif

  // If specified during runtime
  const char* user_cpu_operation = std::getenv(HOROVOD_CONTROLLER);
  if (user_cpu_operation != nullptr) {
    if (strcasecmp(user_cpu_operation, HOROVOD_MPI) == 0) {
      controller = LibType::MPI;
    } else if (strcasecmp(user_cpu_operation, HOROVOD_GLOO) == 0) {
      controller = LibType::GLOO;
    } else {
      throw std::runtime_error("Unsupported controller type, only MPI and Gloo "
                               "are supported");
    }
  }

  LOG(DEBUG) << "Using " << TypeName(controller)
            << " to perform controller operations.";
  return controller;
}

const char* ParseGlooIface() {
  const char* gloo_iface = std::getenv(HOROVOD_GLOO_IFACE);
  if (gloo_iface == nullptr) {
    gloo_iface = GLOO_DEFAULT_IFACE;
  }
  return gloo_iface;
}

void ParseStallInspectorFromEnv(StallInspector& stall_inspector) {
  auto env_value = std::getenv(HOROVOD_STALL_CHECK_DISABLE);
  if (env_value != nullptr && std::strtol(env_value, nullptr, 10) > 0) {
    stall_inspector.SetPerformStallCheck(false);
  }

  env_value = std::getenv(HOROVOD_STALL_CHECK_TIME_SECONDS);
  if (env_value != nullptr) {
    stall_inspector.SetStallWarningTimeSeconds(
        std::strtol(env_value, nullptr, 10));
  }

  env_value = std::getenv(HOROVOD_STALL_SHUTDOWN_TIME_SECONDS);
  if (env_value != nullptr) {
    stall_inspector.SetStallShutdownTimeSeconds(
        std::strtol(env_value, nullptr, 10));
  }
}

void SetBoolFromEnv(const char* env, bool& val, bool value_if_set) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr && std::strtol(env_value, nullptr, 10) > 0) {
    val = value_if_set;
  }
}

bool GetBoolEnvOrDefault(const char* env_variable, bool default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? (bool) std::strtol(env_value, nullptr, 10) : default_value;
}

void SetIntFromEnv(const char* env, int& val) {
  auto env_value = std::getenv(env);
  if (env_value != nullptr) {
    val = std::strtol(env_value, nullptr, 10);
  }
}

int GetIntEnvOrDefault(const char* env_variable, int default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? std::strtol(env_value, nullptr, 10) : default_value;
}

double GetDoubleEnvOrDefault(const char* env_variable, double default_value) {
  auto env_value = std::getenv(env_variable);
  return env_value != nullptr ? std::strtod(env_value, nullptr) : default_value;
}

void SetEnv(const char* env_variable, const char* env_value) {
  setenv(env_variable, env_value, true);
}

} // namespace common
}
