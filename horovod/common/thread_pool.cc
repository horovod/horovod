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

#include "thread_pool.h"

namespace horovod {
namespace common {

void ThreadPool::create(int num_threads) {
  running_ = true;
  threads_.resize(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    threads_[i] = std::thread(&ThreadPool::loop, this);
  }
}

ThreadPool::~ThreadPool() {
  reset();
}

void ThreadPool::execute(std::function<void(void)> f) {
  {
    std::lock_guard<std::mutex> guard(mutex_);
    work_queue_.push(f);
  }
  cond_.notify_one();
}

void ThreadPool::reset() {
  std::unique_lock<std::mutex> lock(mutex_);
  running_ = false;
  cond_.notify_all();
  lock.unlock();

  for (auto& thread: threads_) {
    thread.join();
  }
  threads_.clear();
}

void ThreadPool::loop() {
  while (running_) {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] {return !(running_ && work_queue_.empty());});
    if (!running_) break;

    auto f = work_queue_.front();
    work_queue_.pop();
    lock.unlock();

    f();
  }
}

} // namespace common
} // namespace horovod
