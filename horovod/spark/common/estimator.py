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

import horovod.spark.common._namedtuple_fix

import time

from pyspark.ml import Estimator, Model

from petastorm.spark import SparkDatasetConverter

from horovod.spark.common import util
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.params import EstimatorParams, ModelParams


class HorovodEstimator(Estimator, EstimatorParams):
    def fit(self, df, params=None):
        """Fits the model to the DataFrame.

        Args:
            df: Input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`.
            params: An optional param map that overrides embedded params.
        Returns:
            `HorovodModel` transformer wrapping the trained model.
        """
        return super(HorovodEstimator, self).fit(df, params)

    def fit_on_parquet(self, train_data_path, val_data_path=None, params=None):
        """Trains the model on the given saved Parquet dataset.

        Args:
            train_data_path: String path or list of paths to the training Parquet dataset.
            val_data_path: Optional string path or list of paths to the validation Parquet dataset.
            params: An optional param map that overrides embedded params.

        Returns:
            Trained HorovodModel transformer of the appropriate subclass wrapping the trained model.
        """
        obj = self.copy(params) if params else self
        return obj._fit_on_parquet(train_data_path, val_data_path)

    def _fit_on_parquet(self, train_data_path, val_data_path):
        train_data = SparkDatasetConverter(cache_dir_url=None, file_urls=train_data_path, dataset_size=0)
        val_data = SparkDatasetConverter(cache_dir_url=None, file_urls=val_data_path, dataset_size=0) \
            if val_data_path is not None else None

        backend = self._get_or_create_backend()
        metadata = util.get_dataset_metadata(
            train_data, self.getFeatureCols(), self.getLabelCols(), self.getSampleWeightCol())

        run_id = self._get_or_create_run_id()
        return self._fit_on_prepared_data(backend, metadata, run_id, train_data, val_data)

    def _fit(self, df):
        backend = self._get_or_create_backend()
        with util.prepare_data(df,
                               label_columns=self.getLabelCols(),
                               feature_columns=self.getFeatureCols(),
                               validation=self.getValidation(),
                               sample_weight_col=self.getSampleWeightCol(),
                               compress_sparse=self.getCompressSparseCols()) as (train_data, val_data, metadata):
            self._check_metadata_compatibility(metadata)

            run_id = self._get_or_create_run_id()
            return self._fit_on_prepared_data(backend, metadata, run_id, train_data, val_data)

    def _get_or_create_run_id(self):
        run_id = self.getRunId()
        if run_id is None:
            run_id = self.__class__.__name__ + str(int(time.time()))
        return run_id

    def _get_or_create_backend(self):
        backend = self.getBackend()
        if backend is None:
            backend = SparkBackend(self.getNumProc(), verbose=self.getVerbose())
        elif self.getNumProc() is not None:
            raise ValueError('At most one of parameters "backend" and "num_proc" may be specified')
        return backend

    def _has_checkpoint(self, run_id):
        store = self.getStore()
        last_ckpt_path = store.get_checkpoint_path(run_id)
        return last_ckpt_path is not None and store.exists(last_ckpt_path)


class HorovodModel(Model, ModelParams):
    def transform(self, df, params=None):
        """
        Transforms the input dataset with prediction columns representing model predictions.

        Prediction column names default to <label_column>__output. Override column names
        by calling `transformer.setOutputCols(col_names)`.

        Args:
            df: Input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`.
            params: An optional param map that overrides embedded params.

        Returns:
            Transformed dataset.
        """
        return super(HorovodModel, self).transform(df, params)
