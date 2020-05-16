# Copyright 2020 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


class CommitStateCallbackImpl(object):
    def __init__(self, backend, state, batches_per_commit, *args):
        super(CommitStateCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.state = state
        self.batches_per_commit = batches_per_commit
        self.batches_remaining = batches_per_commit

    def on_batch_end(self, batch, logs=None):
        self.batches_remaining -= 1
        if self.batches_remaining == 0:
            self.state.commit()
            self.batches_remaining = self.batches_per_commit


class UpdateBatchStateCallbackImpl(object):
    def __init__(self, backend, state, *args):
        super(UpdateBatchStateCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.state = state
        self.steps_per_epoch = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.params.get('steps'):
            if self.steps_per_epoch is None:
                self.steps_per_epoch = self.params.get('steps')
            self.params['steps'] = self.steps_per_epoch - self.state.batch

    def on_batch_end(self, batch, logs=None):
        self.state.batch = batch

    def on_epoch_end(self, epoch, logs=None):
        self.state.batch = 0


class UpdateEpochStateCallbackImpl(object):
    def __init__(self, backend, state, *args):
        super(UpdateEpochStateCallbackImpl, self).__init__(*args)
        self.backend = backend
        self.state = state

    def on_epoch_end(self, epoch, logs=None):
        self.state.epoch = epoch
