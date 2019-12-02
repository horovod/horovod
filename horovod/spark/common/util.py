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

from __future__ import absolute_import

import horovod.spark.common._namedtuple_fix

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector, SparseVector, VectorUDT
from pyspark.sql.types import (IntegerType, StringType, FloatType,
                               BinaryType, DoubleType, LongType, BooleanType,
                               ArrayType)
from pyspark.sql.types import from_arrow_type

from horovod.spark.common import cache, constants

_training_cache = cache.TrainingDataCache()


def data_type_to_str(dtype):
    if dtype == VectorUDT:
        return 'Vector'
    elif dtype == IntegerType:
        return 'Int'
    elif dtype == StringType:
        return 'String'
    elif dtype == FloatType:
        return 'Float'
    elif dtype == BinaryType:
        return 'Binary'
    elif dtype == DoubleType:
        return 'Double'
    elif dtype == LongType:
        return 'Long'
    elif dtype == BooleanType:
        return 'Boolean'
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def numpy_type_to_str(dtype):
    if dtype == np.int32:
        return 'Int'
    elif dtype == np.float32:
        return 'Float'
    elif dtype == np.uint8:
        return 'Binary'
    elif dtype == np.float64:
        return 'Double'
    elif dtype == np.int64:
        return 'Long'
    elif dtype == np.bool:
        return 'Boolean'
    else:
        raise ValueError('Cannot convert numpy data type to Spark string: {}'.format(dtype))


def spark_scalar_to_python_type(dtype):
    if dtype == IntegerType:
        return int
    elif dtype == StringType:
        return str
    elif dtype == FloatType:
        return float
    elif dtype == DoubleType:
        return float
    elif dtype == LongType:
        return int
    elif dtype == BooleanType:
        return bool
    elif dtype == BinaryType:
        return bytes
    else:
        raise ValueError('cannot convert Spark data Type {} to native python type'.format(dtype))


def pyarrow_to_spark_data_type(dtype):
    # PySpark will interpret list types as Arrays, but for ML applications we want to default to
    # treating these as DenseVectors.
    if pa.types.is_list(dtype):
        return DenseVector
    return type(from_arrow_type(dtype))


def data_type_to_numpy(dtype):
    if dtype == VectorUDT or dtype == SparseVector or dtype == DenseVector:
        return np.float64
    elif dtype == ArrayType:
        return np.float64
    elif dtype == IntegerType:
        return np.int32
    elif dtype == StringType:
        return np.uint8
    elif dtype == FloatType:
        return np.float32
    elif dtype == BinaryType:
        return np.uint8
    elif dtype == DoubleType:
        return np.float64
    elif dtype == LongType:
        return np.int64
    elif dtype == BooleanType:
        return np.bool
    else:
        raise ValueError('Unrecognized data type: {}'.format(dtype))


def check_shape_compatibility(metadata, feature_columns, label_columns,
                              input_shapes=None, output_shapes=None):
    # Check for model and input type incompatibility. Columns must have the same size
    # (total number of elements) of the corresponding inputs.
    feature_count = len(feature_columns)
    if input_shapes is not None:
        if feature_count != len(input_shapes):
            raise ValueError('Feature column count {features} must equal '
                             'model inputs count {inputs}'
                             .format(features=feature_count, inputs=len(input_shapes)))

    if all(metadata[col]['shape'] for col in feature_columns):
        for idx, col, input_shape in zip(range(feature_count), feature_columns, input_shapes):
            col_size = metadata[col]['shape']
            input_size = abs(np.prod(input_shape))
            if col_size != input_size:
                raise ValueError(
                    'Feature column \'{col}\' with size {feature} must equal that of the '
                    'model input at index {idx} with size {input}'
                    .format(col=col, feature=col_size, idx=idx, input=input_size))

    label_count = len(label_columns)
    if output_shapes is not None and all(metadata[col]['shape'] for col in label_columns):
        if label_count != len(output_shapes):
            raise ValueError('Label column count {labels} must equal '
                             'model outputs count {outputs}'
                             .format(labels=label_count, outputs=len(output_shapes)))

        for idx, col, output_shape in zip(range(label_count), label_columns, output_shapes):
            col_size = metadata[col]['shape']
            output_size = abs(np.prod(output_shape))
            if col_size != output_size:
                raise ValueError('Label column \'{col}\' with size {label} must equal that of the '
                                 'model output at index {idx} with size {output}'
                                 .format(col=col, label=col_size, idx=idx, output=output_size))


def _get_col_info(df):
    """
    Infer the type and shape of all the columns.
    """

    def get_meta(row):
        row_dict = row.asDict()
        row_schema = []
        for col_name, data_col in row_dict.items():
            dtype = type(data_col)
            if dtype == DenseVector:
                # shape and size of dense vector are the same
                shape = data_col.array.shape[0]
                size = shape
            elif dtype == SparseVector:
                # shape is the total size of vector
                shape = data_col.size
                # size is the number of nonzero elements in the sparse vector
                size = data_col.indices.shape[0]
            else:
                shape = 1
                size = 1
            row_schema.append((col_name, (set([dtype]), set([shape]), set([size]))))
        return row_schema

    def merge(x, y):
        dtypes = x[0]
        dtypes.update(y[0])
        shapes = x[1]
        shapes.update(y[1])
        sizes = x[2]
        sizes.update(y[2])
        return dtypes, shapes, sizes

    raw_col_info_list = df.rdd.flatMap(get_meta).reduceByKey(merge).collect()

    all_col_types = {}
    col_shapes = {}
    col_max_sizes = {}

    for col_info in raw_col_info_list:
        col_name = col_info[0]

        all_col_types[col_name] = col_info[1][0]
        col_shapes[col_name] = col_info[1][1]
        col_max_sizes[col_name] = col_info[1][2]

    # all the rows of each columns must have the same shape
    for col in df.schema.names:
        shape_set = col_shapes[col]
        if len(shape_set) != 1:
            raise ValueError(
                'col {col} does not have uniform shapes. shape set: {shapes_set}'.format(col=col,
                                                                                         shapes_set=shape_set))
        col_shapes[col] = shape_set.pop()

    for col in df.schema.names:
        sizes = col_max_sizes[col]
        if len(sizes) > 1 and not (SparseVector in all_col_types[col]):
            raise ValueError(
                "rows of column {col} have varying sizes. This is only allowed if datatype is "
                "SparseVector or a mix of Sparse and DenseVector.".format(col=col))
        col_max_sizes[col] = max(sizes)

    return all_col_types, col_shapes, col_max_sizes


def _get_metadata(df):
    """
    Infer the type and shape of all the columns and determines if what intermedite format they
    need to be converted to in case they are a vector.

    Example return value:
    {
    'col1': {
        'dtype': <type 'float'>,
        'intermediate_format': 'nochange',
        'max_size': 1,
        'shape': 1
        },
     'col2': {
        'dtype': <type 'float'>,
        'intermediate_format': 'nochange',
        'max_size': 1,
        'shape': 1
        },
     'col3': {
        'dtype': <class 'pyspark.ml.linalg.SparseVector'>,
        'intermediate_format': 'custom_sparse_format',
        'max_size': 37,
        'shape': 56
        }
    }
    """
    all_col_types, col_shapes, col_max_sizes = _get_col_info(df)

    metadata = dict()
    for field in df.schema.fields:
        col = field.name
        col_types = all_col_types[col].copy()

        if DenseVector in col_types and SparseVector in col_types:
            # If a col has DenseVector type (whether it is mixed sparse and dense vector or just
            # DenseVector), convert all of the values to dense vector
            is_sparse_vector_only = False
            spark_data_type = DenseVector
            convert_to_target = constants.ARRAY
        elif DenseVector in col_types:
            # If a col has DenseVector type (whether it is mixed sparse and dense vector or just
            # DenseVector), convert all of the values to dense vector
            is_sparse_vector_only = False
            spark_data_type = DenseVector
            convert_to_target = constants.ARRAY
        elif SparseVector in col_types:
            # If a col has only sparse vectors, convert all the data into custom dense vectors
            is_sparse_vector_only = True
            spark_data_type = SparseVector
            convert_to_target = constants.CUSTOM_SPARSE
        else:
            is_sparse_vector_only = False
            spark_data_type = type(field.dataType)
            convert_to_target = constants.NOCHANGE

        # Explanation of the fields in metadata
        #     dtype:
        #
        #     spark_data_type:
        #         The spark data type from dataframe schema: type(field.dataType). If column has
        #         mixed SparseVector and DenseVector we categorize it as DenseVector.
        #
        #     is_sparse_vector_only:
        #         If all the rows in the column were sparse vectors.
        #
        #     shape:
        #         Determines the shape of the data in the spark dataframe. It is useful for sparse
        #         vectors.
        #
        #     intermediate_format:
        #         Specifies if the column need to be converted to a different format so that
        #         petastorm can read it. It can be one of ARRAY, CUSTOM_SPARSE, or NOCHANGE. It is
        #         required because petastorm cannot read DenseVector and SparseVectors. We need to
        #         identify these types and convert them to petastorm compatible type of array.

        metadata[col] = {'spark_data_type': spark_data_type,
                         'is_sparse_vector_only': is_sparse_vector_only,
                         'shape': col_shapes[col],
                         'intermediate_format': convert_to_target,
                         'max_size': col_max_sizes[col]}

    return metadata


def to_petastorm_fn(schema_cols, metadata):
    ARRAY = constants.ARRAY
    CUSTOM_SPARSE = constants.CUSTOM_SPARSE

    # Convert Spark Vectors into arrays so Petastorm can read them
    def to_petastorm(row):
        import numpy as np
        from pyspark import Row

        fields = row.asDict().copy()
        for col in schema_cols:
            col_data = row[col]
            intermediate_format = metadata[col]['intermediate_format']
            if intermediate_format == ARRAY:
                fields[col] = col_data.toArray().tolist()
            elif intermediate_format == CUSTOM_SPARSE:
                # Currently petastorm does not support reading pyspark sparse vector. We put
                # the indices and values into one array. when consuming the data, we re-create
                # the vector from this format.
                size = len(col_data.indices)
                padding_zeros = 2 * (metadata[col]['max_size'] - len(col_data.indices))

                fields[col] = np.concatenate(
                    (np.array([size]), col_data.indices, col_data.values,
                     np.zeros(padding_zeros))).tolist()

        return Row(**fields)

    return to_petastorm


def get_simple_meta_from_parquet(store, label_columns, feature_columns, sample_weight_col):
    train_data_path = store.get_train_data_path()
    validation_data_path = store.get_val_data_path()

    if not store.exists(train_data_path):
        raise ValueError("{} path does not exist on the store".format(train_data_path))

    train_data = pq.ParquetDataset(train_data_path, filesystem=store.get_filesystem())
    schema = train_data.schema.to_arrow_schema()

    train_rows = 0
    total_byte_size = 0
    for piece in train_data.pieces:
        metadata = piece.get_metadata()
        train_rows += metadata.num_rows

        pfile = piece.open()
        file_metadata = pfile.metadata
        for row_group_index in range(file_metadata.num_row_groups):
            row_group = file_metadata.row_group(row_group_index)
            total_byte_size += row_group.total_byte_size

    if train_rows == 0:
        raise ValueError('No rows found in training dataset: {}'.format(train_data_path))

    if total_byte_size == 0:
        raise ValueError('No data found in training dataset: {}'.format(train_data_path))

    if train_rows > total_byte_size:
        raise ValueError('Found {} bytes in {} rows.  Training dataset may be corrupted.'
                         .format(total_byte_size, train_rows))

    val_rows = 0
    if store.exists(validation_data_path):
        val_data = pq.ParquetDataset(validation_data_path, filesystem=store.get_filesystem())
        for piece in val_data.pieces:
            metadata = piece.get_metadata()
            val_rows += metadata.num_rows

    schema_cols = feature_columns + label_columns
    if sample_weight_col:
        schema_cols.append(sample_weight_col)

    metadata = {}
    for col in schema_cols:
        col_schema = schema.field_by_name(col)
        col_info = {
            'spark_data_type': pyarrow_to_spark_data_type(col_schema.type),
            'is_sparse_vector_only': False,
            'shape': None,  # Only used by SparseVector columns
            'intermediate_format': constants.NOCHANGE,
            'max_size': None  # Only used by SparseVector columns
        }
        metadata[col] = col_info

    avg_row_size = total_byte_size / train_rows
    return train_rows, val_rows, metadata, avg_row_size


def prepare_data(num_processes, store, df, label_columns, feature_columns,
                 validation_col=None, validation_split=0.0, sample_weight_col=None,
                 partitions_per_process=10, verbose=False):
    if validation_split and validation_col:
        raise ValueError("can only specify one of validation_col and validation_split")

    num_partitions = num_processes * partitions_per_process
    if verbose:
        print('num_partitions={}'.format(num_partitions))

    train_data_path = store.get_train_data_path()
    val_data_path = store.get_val_data_path()
    if verbose:
        print('train_data_path={}'.format(train_data_path))
        print('val_data_path={}'.format(val_data_path))

    for col in label_columns:
        if col not in df.columns:
            raise ValueError('Label column {} does not exist in this DataFrame'.format(col))

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in set(label_columns)]

    dataframe_hash = df.__hash__()

    global _training_cache
    key = (dataframe_hash, validation_split, validation_col, train_data_path, val_data_path)
    with _training_cache.lock:
        if _training_cache.is_cached(key):
            entry = _training_cache.get(key)
            train_rows = entry.train_rows
            val_rows = entry.val_rows
            metadata = entry.metadata
            avg_row_size = entry.avg_row_size

            if verbose:
                print('using cached dataframes for key: {}'.format(key))
                print('train_rows={}'.format(train_rows))
                print('val_rows={}'.format(val_rows))
        else:
            if verbose:
                print('writing dataframes')

            schema_cols = feature_columns + label_columns
            if sample_weight_col:
                schema_cols.append(sample_weight_col)
            if validation_col:
                schema_cols.append(validation_col)
            df = df[schema_cols]

            metadata = _get_metadata(df)

            to_petastorm = to_petastorm_fn(schema_cols, metadata)
            train_df = df.rdd.map(to_petastorm).toDF()

            val_df = None
            validation_ratio = validation_split
            if validation_split > 0:
                train_df, val_df = train_df.randomSplit([1.0 - validation_split, validation_split])
            elif validation_col:
                val_df = train_df.filter(f.col(validation_col) > 0).drop(validation_col)
                train_df = train_df.filter(f.col(validation_col) == 0).drop(validation_col)

                # Approximate ratio of validation data to training data for proportionate scale
                # of partitions
                timeout_ms = 1000
                confidence = 0.90
                train_rows = train_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
                val_rows = val_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
                validation_ratio = val_rows / (val_rows + train_rows)

            train_partitions = max(int(num_partitions * (1.0 - validation_ratio)),
                                   partitions_per_process)
            if verbose:
                print('train_partitions={}'.format(train_partitions))

            train_df \
                .coalesce(train_partitions) \
                .write \
                .mode('overwrite') \
                .parquet(train_data_path)

            if val_df:
                val_partitions = max(int(num_partitions * validation_ratio),
                                     partitions_per_process)
                if verbose:
                    print('val_partitions={}'.format(val_partitions))

                val_df \
                    .coalesce(val_partitions) \
                    .write \
                    .mode('overwrite') \
                    .parquet(val_data_path)

            train_rows, val_rows, pq_metadata, avg_row_size = get_simple_meta_from_parquet(
                store, label_columns, feature_columns, sample_weight_col)

            if verbose:
                print('train_rows={}'.format(train_rows))
            if val_df:
                if val_rows == 0:
                    raise ValueError(
                        "{validation_col} col does not any validation samples".format(
                            validation_col=validation_col))
                if verbose:
                    print('val_rows={}'.format(val_rows))

            _training_cache.put(key,
                                value=cache.CacheEntry(store, dataframe_hash, validation_split,
                                                       validation_col, train_rows,
                                                       val_rows, metadata, avg_row_size))

    return train_rows, val_rows, metadata, avg_row_size


def clear_training_cache():
    _training_cache.clear()
