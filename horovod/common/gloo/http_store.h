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

#ifndef HOROVOD_GLOO_HTTP_STORE_H
#define HOROVOD_GLOO_HTTP_STORE_H

#include "HTTPRequest.hpp"

#include "gloo_store.h"

namespace horovod {
namespace common {

#define MAX_RETRY_TIMES 3
#define RETRY_WAITING_TIME_MILLSEC 500
#define HTTP_GET_METHOD "GET"
#define HTTP_PUT_METHOD "PUT"
#define HTTP_DELETE_METHOD "DELETE"
#define HTTP_OK 200
#define HTTP_NOT_FOUND 404

class HTTPStore : public GlooStore {
public:
  HTTPStore(const std::string& server_ip, int port, const std::string& scope,
            int rank)
      : rank_(rank) {
    url_prefix_ +=
        "http://" + server_ip + ":" + std::to_string(port) + "/" + scope + "/";
  }

  void set(const std::string& key, const std::vector<char>& data) override;

  std::vector<char> get(const std::string& key) override;

  void wait(const std::vector<std::string>& keys) override {
    wait(keys, Store::kDefaultTimeout);
  }

  void wait(const std::vector<std::string>& keys,
            const std::chrono::milliseconds& timeout) override;

  bool CheckKeys(const std::vector<std::string>& keys);

  void Finalize() override;

protected:
  // Send HTTP request to server, retry if the status code is not 200 (OK) or
  // 404 (Key not found).
  http::Response PerformHTTP(http::Request& request, const std::string& method,
                             const std::string& body);

  // HTTP GET: result is an out parameter for retrieved value for the key.
  // Return a bool representing whether the key is found in the store.
  bool HTTP_GET(const std::string& key, std::vector<char>& result);

  // HTTP PUT: send HTTP PUT request to server with the key and value data.
  // The key is a string and will be embed into the url; the data is
  // the PUT body.
  void HTTP_PUT(const std::string& key, const std::vector<char>& data);

  // HTTP DELETE: send HTTP DELETE request to server, informing the server that
  // this rank has finished.
  void HTTP_DELETE(const std::string& key);

  std::string url_prefix_;
  int rank_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_GLOO_HTTP_STORE_H
