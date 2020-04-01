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

#ifndef HOROVOD_ENV_PARSER_H
#define HOROVOD_ENV_PARSER_H

#include <iostream>

#include "../stall_inspector.h"

namespace horovod {
namespace common {

enum class LibType { MPI = 0, CCL = 1, GLOO = 2 };

std::string TypeName(LibType type);

LibType ParseCPUOpsFromEnv();

LibType ParseControllerOpsFromEnv();

const char* ParseGlooIface();

void ParseStallInspectorFromEnv(StallInspector& stall_inspector);

void SetBoolFromEnv(const char* env, bool& val, bool value_if_set);

bool GetBoolEnvOrDefault(const char* env_variable, bool default_value);

void SetIntFromEnv(const char* env, int& val);

int GetIntEnvOrDefault(const char* env_variable, int default_value);

double GetDoubleEnvOrDefault(const char* env_variable, double default_value);

void SetEnv(const char* env_variable, const char* env_value);

} // namespace common
} // namespace horovod

#endif // HOROVOD_ENV_PARSER_H
