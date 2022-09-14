import argparse
import os
import subprocess
import sys
from packaging import version

import numpy as np

import pyspark
import pyspark.sql.types as T
from pyspark import SparkConf
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
if version.parse(pyspark.__version__) < version.parse('3.0.0'):
    from pyspark.ml.feature import OneHotEncoderEstimator as OneHotEncoder
else:
    from pyspark.ml.feature import OneHotEncoder
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import horovod.spark.keras as hvd
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store

parser = argparse.ArgumentParser(description='Keras Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark master to connect to')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=12,
                    help='number of epochs to train')
parser.add_argument('--work-dir', default='/tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')

if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize SparkSession
    conf = SparkConf().setAppName('keras_spark_mnist').set('spark.sql.shuffle.partitions', '16')
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    # Setup our store for intermediate data
    store = Store.create(args.work_dir)

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
    train_df = model.transform(df)

    # Train/test split
    train_df, test_df = train_df.randomSplit([0.9, 0.1])

    # Disable GPUs when building the model to prevent memory leaks
    if version.parse(tf.__version__) >= version.parse('2.0.0'):
        # See https://github.com/tensorflow/tensorflow/issues/33168
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        keras.backend.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

    # Define the Keras model without any Horovod-specific parameters
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    optimizer = keras.optimizers.Adadelta(1.0)
    loss = keras.losses.categorical_crossentropy

    # Train a Horovod Spark Estimator on the DataFrame
    backend = SparkBackend(num_proc=args.num_proc,
                           stdout=sys.stdout, stderr=sys.stderr,
                           prefix_output_with_timestamp=True)
    keras_estimator = hvd.KerasEstimator(backend=backend,
                                         store=store,
                                         model=model,
                                         optimizer=optimizer,
                                         loss=loss,
                                         metrics=['accuracy'],
                                         feature_cols=['features'],
                                         label_cols=['label_vec'],
                                         batch_size=args.batch_size,
                                         epochs=args.epochs,
                                         random_seed=1,
                                         inmemory_cache_all=True,
                                         verbose=1,
                                         callbacks=[keras.callbacks.TensorBoard(profile_batch=5)])

    keras_model = keras_estimator.fit(train_df).setOutputCols(['label_prob'])

    # Evaluate the model on the held-out test DataFrame
    pred_df = keras_model.transform(test_df)
    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
    evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
    print('Test accuracy:', evaluator.evaluate(pred_df))

    spark.stop()
