import argparse
import os
import subprocess
import sys
from distutils.version import LooseVersion

import numpy as np
import pyspark
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

if LooseVersion(pyspark.__version__) < LooseVersion('3.0.0'):
    from pyspark.ml.feature import OneHotEncoderEstimator as OneHotEncoder
else:
    from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql import Row

import tensorflow as tf
from tensorflow import keras
import io
import h5py
import pyarrow as pa

import horovod.spark
import horovod.tensorflow.keras as hvd
from horovod.spark.common.util import to_petastorm


parser = argparse.ArgumentParser(description='Keras Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--processing-master',
                    help='spark cluster to use for light processing (data preparation & prediction).'
                         'If set to None, uses current default cluster. Cluster should be set up to provide'
                         'one task per CPU core. Example: spark://hostname:7077')
parser.add_argument('--training-master', default='local[2]',
                    help='spark cluster to use for training. If set to None, uses current default cluster. Cluster'
                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode. Example: spark://hostname:7077')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=12,
                    help='number of epochs to train')
parser.add_argument('--work-dir', default='file:///tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')


def serialize_model(model):
    """Serialize model into byte array."""
    bio = io.BytesIO()
    with h5py.File(bio) as f:
        model.save(f)
    return bio.getvalue()


def deserialize_model(model_bytes, load_model_fn):
    """Deserialize model from byte array."""
    bio = io.BytesIO(model_bytes)
    with h5py.File(bio) as f:
        return load_model_fn(f)


if __name__ == '__main__':
    args = parser.parse_args()

    # HDFS driver to use with Petastorm.
    PETASTORM_HDFS_DRIVER = 'libhdfs'

    # ================ #
    # DATA PREPARATION #
    # ================ #

    print('================')
    print('Data preparation')
    print('================')

    # Create Spark session for data preparation.
    conf = SparkConf() \
        .setAppName('Keras Spark MNIST Run Example - Data Prep') \
        .set('spark.sql.shuffle.partitions', '16')
    if args.processing_master:
        conf.setMaster(args.processing_master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Download MNIST dataset
    data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
    libsvm_path = os.path.join(args.data_dir, 'mnist.bz2')
    if not os.path.exists(libsvm_path):
        subprocess.check_output(['wget', data_url, '-O', libsvm_path])

    # Load dataset into a Spark DataFrame
    df = spark.read.format('libsvm') \
        .option('numFeatures', '784') \
        .load(libsvm_path)

    # One-hot encode labels into SparseVectors
    encoder = OneHotEncoder(inputCols=['label'],
                            outputCols=['label_vec'],
                            dropLast=False)
    model = encoder.fit(df)
    train_df = model.transform(df)[['features', 'label_vec']]

    metadata, train_df = to_petastorm(train_df)

    # Train/test split
    train_df, val_df, test_df = train_df.randomSplit([0.8, 0.1, 0.1])

    print('================')
    print('Data frame sizes')
    print('================')
    train_rows, val_rows, test_rows = train_df.count(), val_df.count(), test_df.count()
    print('Training: %d' % train_rows)
    print('Validation: %d' % val_rows)
    print('Test: %d' % test_rows)

    # Save data frames as Parquet files.
    train_df.write.parquet('%s/train_df.parquet' % args.data_dir, mode='overwrite')
    val_df.write.parquet('%s/val_df.parquet' % args.data_dir, mode='overwrite')
    test_df.write.parquet('%s/test_df.parquet' % args.data_dir, mode='overwrite')

    spark.stop()

    # ============== #
    # MODEL TRAINING #
    # ============== #

    print('==============')
    print('Model training')
    print('==============')

    # Disable GPUs when building the model to prevent memory leaks
    if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        # See https://github.com/tensorflow/tensorflow/issues/33168
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

    #dataset = tf.data.Dataset.from_tensor_slices(
    #    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
    #     tf.cast(mnist_labels, tf.int64))
    #)
    #dataset = dataset.repeat().shuffle(10000).batch(batch_size)

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, [3, 3], activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    loss = tf.losses.SparseCategoricalCrossentropy()
    opt = tf.optimizers.Adam(0.001)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt, backward_passes_per_step=1, average_aggregated_gradients=True)

    # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
    # uses hvd.DistributedOptimizer() to compute gradients.
    model.compile(loss=loss,
                  optimizer=opt,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)

    model_bytes = serialize_model(model)


    def train_fn(model_bytes):
        # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
        # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
        # functions like `deserialize_model` which are implemented at the top level.
        # See https://jira.apache.org/jira/browse/ARROW-3346
        pa

        import atexit
        import horovod.tensorflow.keras as hvd
        import os
        from petastorm import make_batch_reader
        from petastorm.tf_utils import make_petastorm_dataset
        import tempfile
        import tensorflow as tf
        import tensorflow.keras.backend as K
        import shutil

        # Horovod: initialize Horovod inside the trainer.
        hvd.init()

        # Horovod: pin GPU to be used to process local rank (one GPU per process)
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        if gpus:
            tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

        # Horovod: restore from checkpoint, use hvd.load_model under the hood.
        model = deserialize_model(model_bytes, hvd.load_model)

        # Horovod: adjust learning rate based on number of processes.
        scaled_lr = K.get_value(model.optimizer.lr) * hvd.size()
        K.set_value(model.optimizer.lr, scaled_lr)

        # Horovod: print summary logs on the first worker.
        verbose = 2 if hvd.rank() == 0 else 0

        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),

            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),

            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=3, verbose=1),
        ]

        # Model checkpoint location.
        ckpt_dir = tempfile.mkdtemp()
        ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
        atexit.register(lambda: shutil.rmtree(ckpt_dir))

        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

        # Make Petastorm readers.
        with make_batch_reader('file://%s/train_df.parquet' % args.data_dir, num_epochs=None,
                               cur_shard=hvd.rank(), shard_count=hvd.size(),
                               hdfs_driver=PETASTORM_HDFS_DRIVER) as train_reader:
            with make_batch_reader('file://%s/val_df.parquet' % args.data_dir, num_epochs=None,
                                   cur_shard=hvd.rank(), shard_count=hvd.size(),
                                   hdfs_driver=PETASTORM_HDFS_DRIVER) as val_reader:
                # Convert readers to tf.data.Dataset.
                train_ds = make_petastorm_dataset(train_reader) \
                    .apply(tf.data.experimental.unbatch()) \
                    .shuffle(int(train_rows / hvd.size())) \
                    .batch(args.batch_size)

                val_ds = make_petastorm_dataset(val_reader) \
                    .apply(tf.data.experimental.unbatch()) \
                    .batch(args.batch_size)

                # Train the model.
                history = model.fit(train_ds,
                                    validation_data=val_ds,
                                    steps_per_epoch=int(train_rows / args.batch_size / hvd.size()),
                                    validation_steps=int(val_rows / args.batch_size / hvd.size()),
                                    callbacks=callbacks,
                                    verbose=verbose,
                                    epochs=args.epochs)

        # Dataset API usage currently displays a wall of errors upon termination.
        # This global model registration ensures clean termination.
        # Tracked in https://github.com/tensorflow/tensorflow/issues/24570
        globals()['_DATASET_FINALIZATION_HACK'] = model

        if hvd.rank() == 0:
            with open(ckpt_file, 'rb') as f:
                return history.history, f.read()

    # Create Spark session for training.
    conf = SparkConf() \
        .setAppName('Keras Spark MNIST Run Example - Training')
    if args.training_master:
        conf.setMaster(args.training_master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Horovod: run training.
    history, best_model_bytes = \
        horovod.spark.run(train_fn, args=(model_bytes,), num_proc=args.num_proc,
                          use_gloo=True,
                          stdout=sys.stdout, stderr=sys.stderr, verbose=2,
                          prefix_output_with_timestamp=True)[0]

    best_val_rmspe = min(history['val_exp_rmspe'])
    print('Best RMSPE: %f' % best_val_rmspe)

    # Write checkpoint.
    with open(args.local_checkpoint_file, 'wb') as f:
        f.write(best_model_bytes)
    print('Written checkpoint to %s' % args.local_checkpoint_file)

    spark.stop()

    # ================ #
    # FINAL PREDICTION #
    # ================ #

    print('================')
    print('Final prediction')
    print('================')

    # Create Spark session for prediction.
    conf = SparkConf() \
        .setAppName('Keras Spark MNIST Run Example - Prediction') \
        .setExecutorEnv('LD_LIBRARY_PATH', os.environ.get('LD_LIBRARY_PATH')) \
        .setExecutorEnv('PATH', os.environ.get('PATH'))
    if args.processing_master:
        conf.setMaster(args.processing_master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    test_df = spark.read.parquet('%s/test_df.parquet' % args.data_dir)
    all_cols = test_df.columns

    def predict_fn(model_bytes):
        def fn(rows):
            import tensorflow as tf
            import tensorflow.keras.backend as K

            # Do not use GPUs for prediction, use single CPU core per task.
            tf.config.experimental.set_visible_devices([], 'GPU')
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)

            # Restore from checkpoint.
            model = deserialize_model(model_bytes, tf.keras.models.load_model)

            # Perform predictions.
            for row in rows:
                fields = row.asDict().copy()
                label_prob = model.predict_on_batch([np.array([row[col]]) for col in all_cols])[0]
                fields['label_pred'] = float(np.argmax(label_prob))
                yield Row(**fields)

        return fn

    # Evaluate the model on the held-out test DataFrame
    pred_df = test_df.rdd.mapPartitions(predict_fn(best_model_bytes)).toDF()
    pred_df.show()
    evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
    print('Test accuracy:', evaluator.evaluate(pred_df))

    spark.stop()
