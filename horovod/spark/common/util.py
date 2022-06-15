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

import contextlib
import os
import time

from multiprocessing.pool import ThreadPool
import pyarrow as pa
import numpy as np
import pyspark.sql.functions as f
from pyspark.ml.linalg import DenseVector, SparseVector, Vector, VectorUDT
from pyspark.sql.types import ArrayType, BinaryType, BooleanType, FloatType, DoubleType, \
    IntegerType, LongType, NullType, StringType
try:
    # Spark 3.0 moved to a pandas submodule
    from pyspark.sql.pandas.types import from_arrow_type
except ImportError:
    from pyspark.sql.types import from_arrow_type

from pyspark.sql import SparkSession

from horovod.runner.common.util import codec, host_hash as hh
from horovod.spark.common import cache, constants

_training_cache = cache.TrainingDataCache()


def host_hash(salt=None):
    """
    Computes this host's host hash by invoking horovod.runner.common.util.host_hash.host_hash.

    Consider environment variable CONTAINER_ID which is present when running Spark via YARN.
    A YARN container does not share memory with other containers on the same host,
    so it must be considered a `host` in the sense of the `host_hash`.

    :param salt: extra information to include in the hash, ignores Falsy values
    :return: host hash
    """
    # turn salt into an array of a single string if given
    salt = [str(salt)] if salt else []

    # We would violate resource allocation if we run all tasks of a host in one container.
    # See [issues 1497](https://github.com/horovod/horovod/issues/1497) for details.
    container = os.environ.get("CONTAINER_ID")
    if container is not None:
        salt.append(container)

    return hh.host_hash(salt='-'.join(salt))


def data_type_to_str(dtype):
    if dtype == VectorUDT or dtype == SparseVector or dtype == DenseVector:
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
        return np.str
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
                              input_shapes=None, output_shapes=None, label_shapes=None):
    # Check for model and input type incompatibility. Columns must have the same size
    # (total number of elements) of the corresponding inputs.
    feature_count = len(feature_columns)
    if input_shapes is not None:
        if feature_count != len(input_shapes):
            raise ValueError('Feature column count {features} must equal '
                             'model inputs count {inputs}'
                             .format(features=feature_count, inputs=len(input_shapes)))

        for idx, col, input_shape in zip(range(feature_count), feature_columns, input_shapes):
            col_size = metadata[col]['shape']
            if col_size is None:
                # When training directly on Parquet, we do not compute shape metadata
                continue

            input_size = abs(np.prod(input_shape))
            if col_size != input_size:
                raise ValueError(
                    'Feature column \'{col}\' with size {feature} must equal that of the '
                    'model input at index {idx} with size {input}'
                    .format(col=col, feature=col_size, idx=idx, input=input_size))

    label_count = len(label_columns)
    if label_shapes is not None and label_count != len(label_shapes):
        raise ValueError('Label column count {labels} must equal '
                         'provided label shapes count {outputs}'
                         .format(labels=label_count, outputs=len(label_shapes)))

    if output_shapes is not None and label_count != len(output_shapes):
        raise ValueError('Label column count {labels} must equal '
                         'model outputs count {outputs}'
                         .format(labels=label_count, outputs=len(output_shapes)))

    def _check_label_cols_size(target_shapes, target_name):
        for idx, col, target_shape in zip(range(label_count), label_columns, target_shapes):
            col_size = metadata[col]['shape']
            if col_size is None:
                # When training directly on Parquet, we do not compute shape metadata
                continue

            target_size = abs(np.prod(target_shape))
            if col_size != target_size:
                raise ValueError('Label column \'{col}\' with size {label} must equal that of the '
                                 '{target_name} shape at index {idx} with size {output}'
                                 .format(col=col, label=col_size, idx=idx, output=target_size,
                                         target_name=target_name))

    if label_shapes is not None:
        _check_label_cols_size(label_shapes, 'label')
    elif output_shapes is not None:
        # Check the label size against the model output shapes only if label_shapes is not provided.
        _check_label_cols_size(output_shapes, 'model output')


def _get_col_info(df):
    """
    Infer the type and shape of all the columns.

    NOTE: This function processes the entire DataFrame, and can therefore be very expensive to run.

    TODO(travis): Only run this if user sets compress_sparse param, otherwise convert all to Array.
    """

    def get_meta(row):
        row_dict = row.asDict()
        row_schema = []
        for col_name, data_col in row_dict.items():
            dtype = type(data_col)
            if isinstance(data_col, DenseVector):
                # shape and size of dense vector are the same
                shape = size = data_col.array.shape[0]
            elif isinstance(data_col, SparseVector):
                # shape is the total size of vector
                shape = data_col.size
                # size is the number of nonzero elements in the sparse vector
                size = data_col.indices.shape[0]
            elif isinstance(data_col, list):
                shape = size = len(data_col)
            else:
                shape = size = 1
            row_schema.append((col_name, ({dtype}, {shape}, {size})))
        return row_schema

    def merge(x, y):
        x_dtypes, x_shapes, x_sizes = x
        y_dtypes, y_shapes, y_sizes = y
        dtypes = x_dtypes | y_dtypes
        shapes = x_shapes | y_shapes
        sizes = x_sizes | y_sizes
        return dtypes, {min(shapes), max(shapes)}, {min(sizes), max(sizes)}

    raw_col_info_list = df.rdd.flatMap(get_meta).reduceByKey(merge).collect()

    all_col_types = {}
    col_shapes = {}
    col_max_sizes = {}

    for col_info in raw_col_info_list:
        col_name, col_meta = col_info
        dtypes, shapes, sizes = col_meta

        all_col_types[col_name] = dtypes
        col_shapes[col_name] = shapes
        col_max_sizes[col_name] = sizes

    for col in df.schema.names:
        # All rows in every column must have the same shape
        shape_set = col_shapes[col]
        if len(shape_set) != 1:
            raise ValueError(
                'Column {col} does not have uniform shape. '
                'shape set: {shapes_set}'.format(col=col, shapes_set=shape_set))
        col_shapes[col] = shape_set.pop()

        # All rows in every column must have the same size unless they have SparseVectors
        sizes = col_max_sizes[col]
        if len(sizes) > 1 and not (SparseVector in all_col_types[col]):
            raise ValueError(
                'Rows of column {col} have varying sizes. This is only allowed if datatype is '
                'SparseVector or a mix of Sparse and DenseVector.'.format(col=col))
        col_max_sizes[col] = max(sizes)

    return all_col_types, col_shapes, col_max_sizes


def _get_metadata(df):
    """
    Infer the type and shape of all the columns and determines if what intermediate format they
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

        if DenseVector in col_types:
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

        converted = {}
        for col in schema_cols:
            col_data = row[col]
            if isinstance(col_data, Vector):
                intermediate_format = metadata[col]['intermediate_format'] if metadata else ARRAY
                if intermediate_format == ARRAY:
                    converted[col] = col_data.toArray().tolist()
                elif intermediate_format == CUSTOM_SPARSE:
                    # Currently petastorm does not support reading pyspark sparse vector. We put
                    # the indices and values into one array. when consuming the data, we re-create
                    # the vector from this format.
                    size = len(col_data.indices)
                    padding_zeros = 2 * (metadata[col]['max_size'] - len(col_data.indices))

                    converted[col] = np.concatenate(
                        (np.array([size]), col_data.indices, col_data.values,
                         np.zeros(padding_zeros))).tolist()

        if converted:
            row = row.asDict().copy()
            row.update(converted)
        return Row(**row)

    return to_petastorm


def _has_vector_column(df):
    for field in df.schema.fields:
        if isinstance(field.dataType, VectorUDT):
            return True
    return False


def _get_dataset_info(dataset, dataset_id, path):
    total_rows = 0
    total_byte_size = 0
    for piece in dataset.pieces:
        metadata = piece.get_metadata()
        total_rows += metadata.num_rows
        for row_group_index in range(metadata.num_row_groups):
            row_group = metadata.row_group(row_group_index)
            total_byte_size += row_group.total_byte_size

    if total_rows == 0:
        raise ValueError('No rows found in {} dataset: {}'.format(dataset_id, path))

    if total_byte_size == 0:
        raise ValueError('No data found in {} dataset: {}'.format(dataset_id, path))

    if total_rows > total_byte_size:
        raise ValueError('Found {} bytes in {} rows; {} dataset may be corrupted.'
                         .format(total_byte_size, total_rows, dataset_id))

    return total_rows, total_byte_size


def _save_meta_to_fs(fs, path, schema, rows, total_byte_size):
    with fs.open(path, 'wb') as train_meta_file:
        serialized_content = codec.dumps_base64(dict(schema=schema,
                                                     rows=rows,
                                                     total_byte_size=total_byte_size))
        train_meta_file.write(serialized_content.encode('utf-8'))


def _load_metadata_from_fs(fs, path):
    with fs.open(path, 'rb') as train_meta_file:
        meta = train_meta_file.read()
        meta = codec.loads_base64(meta.decode())
        data_schema = meta['schema']
        rows = meta['rows']
        total_byte_size = meta['total_byte_size']

    return data_schema, rows, total_byte_size


def get_simple_meta_from_parquet(store, label_columns, feature_columns, sample_weight_col,
                                 dataset_idx=None):
    train_data_path = store.get_train_data_path(dataset_idx)
    validation_data_path = store.get_val_data_path(dataset_idx)

    if not store.exists(train_data_path):
        raise ValueError("{} path does not exist in the store".format(train_data_path))

    train_data_meta_path = store.get_data_metadata_path(train_data_path)
    val_data_meta_path = store.get_data_metadata_path(validation_data_path)
    fs = store.fs

    schema_cols = feature_columns + label_columns
    if sample_weight_col:
        schema_cols.append(sample_weight_col)

    def make_metadata_dictionary(_train_data_schema):
        _metadata = {}
        for col in schema_cols:
            col_schema = _train_data_schema.field_by_name(col)
            col_info = {
                'spark_data_type': pyarrow_to_spark_data_type(col_schema.type),
                'is_sparse_vector_only': False,
                'shape': None,  # Only used by SparseVector columns
                'intermediate_format': constants.NOCHANGE,
                'max_size': None  # Only used by SparseVector columns
            }
            _metadata[col] = col_info

        _avg_row_size = train_data_total_byte_size / train_rows
        return _metadata, _avg_row_size

    # In the try block we try to read the data metadata from the cached metadata in the store. If
    # anything goes wrong, we will ignore the cache and create the metadata from data.
    try:
        if store.exists(train_data_meta_path) and \
                (store.exists(val_data_meta_path) or not store.exists(validation_data_path)):
            train_data_schema, train_rows, train_data_total_byte_size = \
                _load_metadata_from_fs(fs, train_data_meta_path)
            metadata, avg_row_size = make_metadata_dictionary(train_data_schema)

            val_rows = 0
            if store.exists(val_data_meta_path):
                val_data_schema, val_rows, val_data_total_byte_size = _load_metadata_from_fs(fs,
                                                                                             val_data_meta_path)

            return train_rows, val_rows, metadata, avg_row_size
    except Exception as ex:
        print(ex)

    train_data = store.get_parquet_dataset(train_data_path)
    train_data_schema = train_data.schema.to_arrow_schema()
    train_rows, train_data_total_byte_size = _get_dataset_info(train_data, 'training',
                                                               train_data_path)

    # Write train metadata to filesystem
    _save_meta_to_fs(fs, train_data_meta_path, train_data_schema, train_rows,
                     train_data_total_byte_size)

    val_rows = 0
    if store.exists(validation_data_path):
        val_data = store.get_parquet_dataset(validation_data_path)
        val_data_schema = val_data.schema.to_arrow_schema()
        val_rows, val_data_total_byte_size = _get_dataset_info(val_data, 'validation',
                                                               validation_data_path)

        # Write validation metadata to filesystem
        _save_meta_to_fs(fs, val_data_meta_path, val_data_schema, val_rows,
                         val_data_total_byte_size)
    metadata, avg_row_size = make_metadata_dictionary(train_data_schema)
    return train_rows, val_rows, metadata, avg_row_size


def _train_val_split(df, validation):
    train_df = df
    val_df = None
    validation_ratio = 0.0

    if isinstance(validation, float) and validation > 0:
        train_df, val_df = train_df.randomSplit([1.0 - validation, validation])
        validation_ratio = validation
    elif isinstance(validation, str):
        dtype = [field.dataType for field in df.schema.fields if field.name == validation][0]
        bool_dtype = isinstance(dtype, BooleanType)
        val_df = train_df.filter(
            f.col(validation) if bool_dtype else f.col(validation) > 0).drop(validation)
        train_df = train_df.filter(
            ~f.col(validation) if bool_dtype else f.col(validation) == 0).drop(validation)

        # Approximate ratio of validation data to training data for proportionate scale
        # of partitions
        timeout_ms = 1000
        confidence = 0.90
        train_rows = train_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
        val_rows = val_df.rdd.countApprox(timeout=timeout_ms, confidence=confidence)
        validation_ratio = val_rows / (val_rows + train_rows)
    elif validation:
        raise ValueError('Unrecognized validation type: {}'.format(type(validation)))

    return train_df, val_df, validation_ratio


_DATABRICKS_FILE_AVAILABILITY_WAIT_TIMEOUT_SECS = \
    int(os.environ.get('DATABRICKS_FILE_AVAILABILITY_WAIT_TIMEOUT_SECS', '30'))


_DATABRICKS_FILE_AVAILABILITY_CHECK_INTERVAL_SECS = \
    float(os.environ.get('DATABRICKS_FILE_AVAILABILITY_CHECK_INTERVAL_SECS', '0.1'))


def _wait_file_available_on_dbfs(store, url_list):
    """
    On databricks runtime, Waiting about DATABRICKS_FILE_AVAILABILITY_WAIT_TIMEOUT_SECS seconds
    (default 30 seconds) to make sure all files are available for reading.
    This is because Databricks filesystem backend storage such as S3 which only providing
    eventually consistency.
    """
    # Import LocalStore here to avoid circular import
    from horovod.spark.common.store import LocalStore
    if isinstance(store, LocalStore):
        return

    if not is_databricks():
        return

    def wait_for_file(path):
        end_time = time.time() + _DATABRICKS_FILE_AVAILABILITY_WAIT_TIMEOUT_SECS
        while time.time() < end_time:
            if store.exists(path):
                return True
            time.sleep(_DATABRICKS_FILE_AVAILABILITY_CHECK_INTERVAL_SECS)
        return False

    if len(url_list) == 0:
        raise ValueError('Input url_list argument is empty.')

    pool = ThreadPool(min(len(url_list), 64))
    try:
        results = pool.map(wait_for_file, url_list)
        failed_list = [url for url, result in zip(url_list, results) if not result]
        if failed_list:
            raise TimeoutError('Timeout while waiting for all files to appear at urls {failed_list}.'
                               .format(failed_list=','.join(failed_list)))
    finally:
        pool.close()
        pool.join()


def _get_spark_df_saved_file_list(saved_path):
    spark_session = SparkSession.builder.getOrCreate()
    return list(spark_session.read.parquet(saved_path)._jdf.inputFiles())


def _get_or_create_dataset(key, store, df, feature_columns, label_columns,
                           validation, sample_weight_col, compress_sparse,
                           num_partitions, num_processes, verbose):
    with _training_cache.lock:
        if _training_cache.is_cached(key, store):
            dataset_idx = _training_cache.get_dataset(key)
            train_rows, val_rows, metadata, avg_row_size = _training_cache.get_dataset_properties(dataset_idx)
            train_data_path = store.get_train_data_path(dataset_idx)
            val_data_path = store.get_val_data_path(dataset_idx)
            if verbose:
                print('using cached dataframes for key: {}'.format(key))
                print('train_data_path={}'.format(train_data_path))
                print('train_rows={}'.format(train_rows))
                print('val_data_path={}'.format(val_data_path))
                print('val_rows={}'.format(val_rows))
        else:
            dataset_idx = _training_cache.next_dataset_index(key)
            train_data_path = store.get_train_data_path(dataset_idx)
            val_data_path = store.get_val_data_path(dataset_idx)
            if verbose:
                print('writing dataframes')
                print('train_data_path={}'.format(train_data_path))
                print('val_data_path={}'.format(val_data_path))

            schema_cols = feature_columns + label_columns
            if sample_weight_col:
                schema_cols.append(sample_weight_col)
            if isinstance(validation, str):
                schema_cols.append(validation)
            df = df[schema_cols]

            metadata = None
            if _has_vector_column(df):
                if compress_sparse:
                    metadata = _get_metadata(df)
                to_petastorm = to_petastorm_fn(schema_cols, metadata)
                df = df.rdd.map(to_petastorm).toDF()

            train_df, val_df, validation_ratio = _train_val_split(df, validation)

            train_partitions = max(int(num_partitions * (1.0 - validation_ratio)),
                                   num_processes)
            if verbose:
                print('train_partitions={}'.format(train_partitions))

            train_df \
                .coalesce(train_partitions) \
                .write \
                .mode('overwrite') \
                .parquet(train_data_path)

            saved_file_list = _get_spark_df_saved_file_list(train_data_path)

            if val_df:
                val_partitions = max(int(num_partitions * validation_ratio),
                                     num_processes)
                if verbose:
                    print('val_partitions={}'.format(val_partitions))

                val_df \
                    .coalesce(val_partitions) \
                    .write \
                    .mode('overwrite') \
                    .parquet(val_data_path)

                saved_file_list += _get_spark_df_saved_file_list(val_data_path)

            try:
                _wait_file_available_on_dbfs(store, saved_file_list)
            except TimeoutError as e:
                err_msg = 'Timeout while waiting for all parquet-store files to appear, Please ' \
                          'check whether these files were saved successfully when materializing ' \
                          'dataframe. Internal Error: {e}'.format(e=str(e))
                raise RuntimeError(err_msg)

            train_rows, val_rows, pq_metadata, avg_row_size = get_simple_meta_from_parquet(
                store, label_columns, feature_columns, sample_weight_col, dataset_idx)

            if verbose:
                print('train_rows={}'.format(train_rows))
            if val_df:
                if val_rows == 0:
                    raise ValueError(
                        'Validation DataFrame is empty with validation param: {}'
                        .format(validation))
                if verbose:
                    print('val_rows={}'.format(val_rows))

            metadata = metadata or pq_metadata
            _training_cache.set_dataset_properties(
                dataset_idx, (train_rows, val_rows, metadata, avg_row_size))
        return dataset_idx


def check_validation(validation, df=None):
    if validation:
        if isinstance(validation, float):
            if validation < 0 or validation >= 1:
                raise ValueError('Validation split {} must be in the range: [0, 1)'
                                 .format(validation))
        elif isinstance(validation, str):
            if df is not None and validation not in df.columns:
                raise ValueError('Validation column {} does not exist in the DataFrame'
                                 .format(validation))
        else:
            raise ValueError('Param validation must be of type "float" or "str", found: {}'
                             .format(type(validation)))


@contextlib.contextmanager
def prepare_data(num_processes, store, df, label_columns, feature_columns,
                 validation=None, sample_weight_col=None, compress_sparse=False,
                 partitions_per_process=10, verbose=0):
    check_validation(validation, df=df)
    if num_processes <= 0 or partitions_per_process <= 0:
        raise ValueError('num_proc={} and partitions_per_process={} must both be > 0'
                         .format(num_processes, partitions_per_process))

    if not label_columns:
        raise ValueError('Parameter label_columns cannot be None or empty')

    num_partitions = num_processes * partitions_per_process
    if verbose:
        print('num_partitions={}'.format(num_partitions))

    for col in label_columns:
        if col not in df.columns:
            raise ValueError('Label column {} does not exist in the DataFrame'.format(col))

    if feature_columns is None:
        feature_columns = [col for col in df.columns if col not in set(label_columns)]
    else:
        for col in feature_columns:
            if col not in df.columns:
                raise ValueError('Feature column {} does not exist in the DataFrame'.format(col))

    key = _training_cache.create_key(df, store, validation)
    with _training_cache.use_key(key):
        dataset_idx = _get_or_create_dataset(key, store, df, feature_columns, label_columns,
                                             validation, sample_weight_col, compress_sparse,
                                             num_partitions, num_processes, verbose)
        yield dataset_idx


def get_dataset_properties(dataset_idx):
    return _training_cache.get_dataset_properties(dataset_idx)


def clear_training_cache():
    _training_cache.clear()


def to_list(var, length):
    if var is None:
        return None

    if not isinstance(var, list):
        var = [var]

    # If var has only one element, pad it to match the given length.
    if len(var) == 1:
        var = [var[0] for _ in range(length)]
    else:
        count = len(var)
        if count != length:
            raise ValueError(f'List {var} must be length {length} (found: {count})')
    return var


def _get_assigned_gpu_or_default(default):
    from horovod.spark.task import get_available_devices
    available_devices = get_available_devices()
    if available_devices:
        # if GPU-aware scheduling is available, pin to the assigned GPU index
        return int(available_devices[0])
    else:
        # pin to default GPU index (local rank)
        return default


def is_databricks():
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_output_cols(input_df_schema, output_cols):
    final_output_cols = [f.name for f in input_df_schema.fields]
    return final_output_cols + output_cols


def get_spark_df_output_schema(input_df_schema, label_cols, output_cols, metadata):
    if len(label_cols) != len(output_cols):
        raise Exception('Number of output columns must equal to number of label columns.')

    import copy
    from pyspark.sql.types import StructType, StructField
    from pyspark.ml.linalg import VectorUDT

    ## output col name should not be the same as any data in existing df
    input_df_col_set = {f.name for f in input_df_schema.fields}
    for col in output_cols:
        if col in input_df_col_set:
            raise ValueError("Output col '{}' exists in original df schema: {}".format(col, input_df_col_set))

    # assuming the label_cols and output_cols are 1:1 mapping.
    output_fields = copy.deepcopy(input_df_schema.fields)
    for label_col, output_col in zip(label_cols, output_cols):
        col_type = metadata[label_col]['spark_data_type']
        if col_type == DenseVector or col_type == SparseVector:
            col_type = VectorUDT

        field = StructField(output_col, col_type(), True)
        output_fields.append(field)

    # Final output schema contains both input and output fields
    output_schema = StructType(output_fields)

    return output_schema

def _set_mp_start_method(method, verbose):
    import multiprocessing as mp
    supported_values = mp.get_all_start_methods()
    if method not in supported_values:
        raise ValueError(f"Multiprocessing start method {method} is not supported on this system. "
                         f"Supported values are : {supported_values}")
    mp.set_start_method(method)
    if verbose:
        print(f'Multiprocessing start method has been set to {mp.get_start_method()}')
