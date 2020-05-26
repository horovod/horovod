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

from pyspark.ml import Estimator, Model

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

    def fit_on_parquet(self, params=None, dataset_idx=None):
        """Trains the model on a saved Parquet file at `store.get_train_path()`.

        Args:
            params: An optional param map that overrides embedded params.

        Returns:
            Trained HorovodModel transformer of the appropriate subclass wrapping the trained model.
        """
        if params:
            return self.copy(params)._fit_on_parquet(dataset_idx=dataset_idx)
        return self._fit_on_parquet(dataset_idx=dataset_idx)

    def _fit_on_parquet(self, dataset_idx=None):
        backend = self._get_or_create_backend()
        store = self.getStore()
        label_columns = self.getLabelCols()
        feature_columns = self.getFeatureCols()
        sample_weight_col = self.getSampleWeightCol()

        train_rows, val_rows, metadata, avg_row_size = \
            util.get_simple_meta_from_parquet(store,
                                              label_columns=label_columns,
                                              feature_columns=feature_columns,
                                              sample_weight_col=sample_weight_col,
                                              dataset_idx=dataset_idx)

        return self._fit_on_prepared_data(backend, train_rows, val_rows, metadata, avg_row_size, dataset_idx)

    def _fit(self, df):
        backend = self._get_or_create_backend()
        with util.prepare_data(backend.num_processes(),
                               self.getStore(),
                               df,
                               label_columns=self.getLabelCols(),
                               feature_columns=self.getFeatureCols(),
                               validation=self.getValidation(),
                               sample_weight_col=self.getSampleWeightCol(),
                               compress_sparse=self.getCompressSparseCols(),
                               partitions_per_process=self.getPartitionsPerProcess(),
                               verbose=self.getVerbose()) as dataset_idx:
            train_rows, val_rows, metadata, avg_row_size = util.get_dataset_properties(dataset_idx)
            self._check_metadata_compatibility(metadata)
            return self._fit_on_prepared_data(
                backend, train_rows, val_rows, metadata, avg_row_size, dataset_idx)

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
