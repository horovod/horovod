# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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

import json
import os
import time

from pyspark.ml.util import DefaultParamsWriter, DefaultParamsReader


class HorovodParamsWriter(DefaultParamsWriter):
    @staticmethod
    def saveMetadata(instance, path, sc, extraMetadata=None, paramMap=None,
                     param_serializer_fn=None):
        metadata_path = os.path.join(path, "metadata")
        metadata_json = HorovodParamsWriter. \
            _get_metadata_to_save(instance,
                                  sc,
                                  extraMetadata,
                                  paramMap,
                                  param_serializer_fn)

        # save mode by store as spark has 2G limitation on file size.
        if hasattr(instance, 'getStore') and instance.getStore() is not None:
            instance.getStore().write_text(metadata_path, metadata_json)
        else:
            sc.parallelize([metadata_json], 1).saveAsTextFile(metadata_path)

    @staticmethod
    def _get_metadata_to_save(instance, sc, extra_metadata=None, param_map=None,
                              param_serializer_fn=None):
        uid = instance.uid
        cls = instance.__module__ + '.' + instance.__class__.__name__

        # User-supplied param values
        params = instance._paramMap
        json_params = {}
        if param_map is not None:
            json_params = param_map
        else:
            for p, param_val in params.items():
                # If param is not json serializable, convert it into serializable object
                json_params[p.name] = param_serializer_fn(p.name, param_val)

        # Default param values
        json_default_params = {}
        for p, param_val in instance._defaultParamMap.items():
            json_default_params[p.name] = param_serializer_fn(p.name,
                                                              param_val)

        basic_metadata = {"class": cls, "timestamp": int(round(time.time() * 1000)),
                          "sparkVersion": sc.version, "uid": uid, "paramMap": json_params,
                          "defaultParamMap": json_default_params}
        if extra_metadata is not None:
            basic_metadata.update(extra_metadata)
        return json.dumps(basic_metadata, separators=[',', ':'])


class HorovodParamsReader(DefaultParamsReader):
    def load(self, path):
        metadata = DefaultParamsReader.loadMetadata(path, self.sc)
        metadata['paramMap'] = self._deserialize_dict(metadata['paramMap'])
        metadata['defaultParamMap'] = self._deserialize_dict(metadata['defaultParamMap'])

        py_type = DefaultParamsReader._DefaultParamsReader__get_class(metadata['class'])
        instance = py_type()
        instance._resetUid(metadata['uid'])
        DefaultParamsReader.getAndSetParams(instance, metadata)
        return instance
