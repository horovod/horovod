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

#include "http_rendezvous.h"

#include <cstring>
#include <iostream>
#include <istream>
#include <ostream>
#include <string>
#include <thread>

#include "../logging.h"
#include "gloo/common/error.h"

namespace horovod {
namespace common {

HTTPStore::~HTTPStore() { HTTPDELETE(std::to_string(rank_)); }

void HTTPStore::set(const std::string& key, const std::vector<char>& data) {
  HTTPPUT(key, data);
}

std::vector<char> HTTPStore::get(const std::string& key) {
  std::vector<char> result;
  HTTPGET(key, result);
  return result;
}

void HTTPStore::wait(const std::vector<std::string>& keys,
                     const std::chrono::milliseconds& timeout) {
  const auto start = std::chrono::steady_clock::now();

  while (!CheckKeys(keys)) {
    const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::steady_clock::now() - start);
    if (timeout != gloo::kNoTimeout && elapsed > timeout) {
      GLOO_THROW_IO_EXCEPTION(GLOO_ERROR_MSG("Wait timeout for key(s): ",
                                             ::gloo::MakeString(keys)));
    }
    /* sleep override */
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

bool HTTPStore::CheckKeys(const std::vector<std::string>& keys) {
  std::vector<char> result;
  for (const auto& key : keys) {
    if (!HTTPGET(key, result)) {
      return false;
    }
  }
  return true;
}

// Perform http request to rendezvous server with retry logic
http::Response HTTPStore::PerformHTTP(http::Request& request,
                                      const std::string& method = HTTP_GET,
                                      const std::string& body = "") {
  int retry_cnt = 0;
  while (retry_cnt < MAX_RETRY_TIME) {
    try {
      http::Response response = request.send(method, body);
      if (response.status != HTTP_OK && response.status != HTTP_NOT_FOUND) {
        LOG(WARNING) << "HTTP response not OK, got" << response.status;
      } else {
        return response;
      }
    } catch (std::exception& e) {
      LOG(DEBUG) << "Exception: " << e.what();
    }

    retry_cnt++;
    if (retry_cnt >= 3) {
      LOG(ERROR) << "HTTP GET request failed too many times, aborting. See "
                    "exception message above.";
      throw std::runtime_error("HTTP request failed.");
    }

    // sleep for 500ms before another try.
    std::this_thread::sleep_for(
        std::chrono::milliseconds(RETRY_WAITING_TIME_MILLSEC));
  }

  return http::Response();
}

bool HTTPStore::HTTPGET(const std::string& key, std::vector<char>& result) {
  std::string url = "http://" + server_ip_ + ":" +
                    std::to_string(server_port_) + "/" + scope_ + "/" + key;
  LOG(DEBUG) << "Send GET request to " << url;
  http::Request request(url);

  http::Response response = PerformHTTP(request, HTTP_GET);

  // If the key is not present, return false.
  if (response.status == 404) {
    return false;
  } else {
    result.clear();
    result.insert(result.begin(), response.body.begin(), response.body.end());
    return true;
  }
}

void HTTPStore::HTTPPUT(const std::string& key, const std::vector<char>& data) {
  std::string url = "http://" + server_ip_ + ":" +
                    std::to_string(server_port_) + "/" + scope_ + "/" + key;
  LOG(DEBUG) << "Send PUT request to " << url;
  http::Request request(url);

  std::string body;
  body.insert(body.size(), data.data(), data.size());

  http::Response response = PerformHTTP(request, HTTP_PUT, body);
}

void HTTPStore::HTTPDELETE(const std::string& key) {
  std::string url = "http://" + server_ip_ + ":" +
                    std::to_string(server_port_) + "/" + scope_ + "/" + key;
  LOG(DEBUG) << "Send GET request to " << url;
  http::Request request(url);
  http::Response response = PerformHTTP(request, HTTP_DELETE);
}

} // namespace common
} // namespace horovod
