// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef HOROVOD_THREAD_POOL_H
#define HOROVOD_THREAD_POOL_H

#include <condition_variable>
#include <functional>
#include <queue>
#include <thread>
#include <vector>

namespace horovod {
namespace common {
class ThreadPool {
  public:
    ~ThreadPool();
    void create(int num_threads);
    void reset();
    void execute(std::function<void(void)> f);

  private:
    void loop();
    bool running_;
    std::queue<std::function<void(void)>> work_queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::vector<std::thread> threads_;
};

} // namespace common
} // namespace horovod

#endif // HOROVOD_THREAD_POOL_H
