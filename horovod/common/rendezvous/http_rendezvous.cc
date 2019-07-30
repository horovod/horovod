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

HTTPStore::HTTPStore(const char* server_addr, int port, const char* scope,
                     int rank) {
  server_ip_ = std::string(server_addr);
  server_port_ = port;
  scope_ = scope;
  rank_ = rank;
}


HTTPStore::~HTTPStore() {
  PerformHTTP(std::string(), std::vector<char>(), FINALIZE);
}

// Performing a single http request using given args,
// throw runtime_errors when exceptions occur. Does not have the retry logic.
std::vector<char> HTTPStore::PerformSingleHTTP(const std::string& key,
                                               const std::vector<char>& data,
                                               Type type) {
  std::string url = "http://" + server_ip_ + ":" +
                    std::to_string(server_port_) + "/" + scope_ + "/" +
                    GetEndpoint(type);
  LOG(DEBUG) << "Send request to " << url;
  http::Request request(url);

  // pass parameters as a map
  std::vector<std::string> headers;
  headers.push_back(GenerateHeaderLine("Key-Length", key.size()));
  headers.push_back(GenerateHeaderLine("Value-Length", data.size()));
  headers.push_back(GenerateHeaderLine("Rank", rank_));
  headers.emplace_back("Content-Type: application/octet-stream");

  std::string body(key);
  body.insert(body.size(), data.data(), data.size());

  http::Response response = request.send("POST", body, headers);
  if (response.status != 200){
    std::string msg("HTTP response not OK, got ");
    msg += std::to_string(response.status);
    throw std::runtime_error(msg);
  }

  if (response.body.size() == 0){
    LOG(DEBUG) << "Receive empty body, with status code " << response.status;
  }
  std::vector<char> result(response.body.begin(), response.body.end());

  LOG(DEBUG) << "Got response with length " << response.body.size();
  return result;
}

void HTTPStore::set(const std::string& key, const std::vector<char>& data) {
  PerformHTTP(key, data, SET);
}

std::vector<char> HTTPStore::get(const std::string& key) {
  return PerformHTTP(key, std::vector<char>(), GET);
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
  for (const auto& key : keys) {
    if (PerformHTTP(key, std::vector<char>(), GET).empty()) {
      return false;
    }
  }
  return true;
}

// Perform http request to rendezvous server with retry logic
std::vector<char> HTTPStore::PerformHTTP(const std::string& key,
                                         const std::vector<char>& data,
                                         Type type) {
  int retry_cnt = 0;

  while (retry_cnt < 3) {
    try {
      return PerformSingleHTTP(key, data, type);
    } catch (std::runtime_error& e) {
      retry_cnt++;
      LOG(DEBUG) << "Exception: " << e.what();
      if (retry_cnt >= 3) {
        std::cout << "HTTP request failed too many times, aborting. See "
                     "exception message above.";
        throw e;
      }
    }
  }
  return std::vector<char>();
}

std::string HTTPStore::GenerateHeaderLine(const char* key, int value) {
  return std::string(key) + ": " + std::to_string(value);
}

std::string HTTPStore::GetEndpoint(Type type) {
  switch (type){
  case GET:
    return std::string("get");
  case SET:
    return std::string("set");
  case FINALIZE:
    return std::string("finalize");
  }
}

} // namespace common
} // namespace horovod
