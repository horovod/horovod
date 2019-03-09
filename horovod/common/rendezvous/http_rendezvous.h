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

#ifndef HOROVOD_RENDEZVOUS_HTTP_RENDEZVOUS_H_
#define HOROVOD_RENDEZVOUS_HTTP_RENDEZVOUS_H_

#include "gloo/rendezvous/store.h"
#include "HTTPRequest.hpp"

namespace horovod {
namespace common {

class HTTPStore : public gloo::rendezvous::Store {
public:
  HTTPStore(const char* server_addr, int port, const char* scope, int rank);

  void set(const std::string& key, const std::vector<char>& data) override;

  std::vector<char> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override {
    wait(keys, Store::kDefaultTimeout);
  }

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

  bool CheckKeys(const std::vector<std::string>& keys);

protected:
  std::vector<char> PerformHTTP(const std::string& key,
                                const std::vector<char>& data, bool is_get);

  std::vector<char> PerformSingleHTTP(const std::string& key,
                                      const std::vector<char>& data,
                                      bool is_get);

  std::string GenerateHeaderLine(const char* key, int value);

protected:
  std::string server_ip_;
  int server_port_;
  std::string scope_;
  int rank_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_RENDEZVOUS_HTTP_RENDEZVOUS_H_
