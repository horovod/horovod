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

# Spark PyTorch Lightning tests conflict with Tensorflow 2.5.x and 2.6.x: https://github.com/horovod/horovod/pull/3263
try:
    # tensorflow has to be imported BEFORE pytorch_lightning, otherwise we see the segfault right away
    import tensorflow as tf
    from packaging import version
    if version.parse('2.5.0') <= version.parse(tf.__version__) < version.parse('2.7.0'):
        print('Skipping test as Pytorch Lightning conflicts with present Tensorflow 2.6.x', file=sys.stderr)
        sys.exit(0)
except ImportError:
    pass

from pytorch_lightning import LightningModule

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import horovod.spark.lightning as hvd
from horovod.spark.lightning.estimator import MIN_PL_VERSION
from horovod.spark.common.backend import SparkBackend
from horovod.spark.common.store import Store

parser = argparse.ArgumentParser(description='PyTorch Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark master to connect to')
parser.add_argument('--num-proc', type=int, default=2,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=12,
                    help='number of epochs to train')
parser.add_argument('--work-dir', default='/tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')
parser.add_argument('--data-dir', default='/tmp',
                    help='location of the training dataset in the local filesystem (will be downloaded if needed)')
parser.add_argument('--enable-profiler', action='store_true',
                    help='Enable profiler')


def train_model(args):
    # do not run this test for pytorch lightning below min supported verson
    import pytorch_lightning as pl
    if version.parse(pl.__version__) < version.parse(MIN_PL_VERSION):
        print("Skip test for pytorch_ligthning=={}, min support version is {}".format(pl.__version__, MIN_PL_VERSION))
        return

    # Initialize SparkSession
    conf = SparkConf().setAppName('pytorch_spark_mnist').set('spark.sql.shuffle.partitions', '16')
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

    # Define the PyTorch model without any Horovod-specific parameters
    class Net(LightningModule):
        def __init__(self, kernel_size):
            super(Net, self).__init__()
            # The Lightning checkpoint also saves the arguments passed into the LightningModule init
            # under the "hyper_parameters" key in the checkpoint.
            self.save_hyperparameters()
            self.conv1 = nn.Conv2d(1, 10, kernel_size=kernel_size)
            self.conv2 = nn.Conv2d(10, 20, kernel_size=kernel_size)
            self.conv2_drop = nn.Dropout2d()
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = x.float().reshape((-1, 1, 28, 28))
            x = F.relu(F.max_pool2d(self.conv1(x), 2))
            x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
            x = x.view(-1, 320)
            x = F.relu(self.fc1(x))
            x = F.dropout(x, training=self.training)
            x = self.fc2(x)
            return F.log_softmax(x, -1)

        def configure_optimizers(self):
            return optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

        def training_step(self, batch, batch_idx):
            if batch_idx == 0:
                print(f"training data batch size: {batch['label'].shape}")
            x, y = batch['features'], batch['label']
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y.long())
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            if batch_idx == 0:
                print(f"validation data batch size: {batch['label'].shape}")
            x, y = batch['features'], batch['label']
            y_hat = self(x)
            loss = F.nll_loss(y_hat, y.long())
            self.log('val_loss', loss)

        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean() if len(outputs) > 0 else float('inf')
            self.log('avg_val_loss', avg_loss)

    model = Net(5)

    # Train a Horovod Spark Estimator on the DataFrame
    backend = SparkBackend(num_proc=args.num_proc,
                           stdout=sys.stdout, stderr=sys.stderr,
                           prefix_output_with_timestamp=True)

    from pytorch_lightning.callbacks import Callback

    epochs = args.epochs

    class MyDummyCallback(Callback):
        def __init__(self):
            self.epcoh_end_counter = 0
            self.train_epcoh_end_counter = 0
            self.validation_epoch_end_counter = 0

        def on_init_start(self, trainer):
            print('Starting to init trainer!')

        def on_init_end(self, trainer):
            print('Trainer is initialized.')

        def on_epoch_end(self, trainer, model):
            print('A train or eval epoch ended.')
            self.epcoh_end_counter += 1

        def on_train_epoch_end(self, trainer, model, unused=None):
            print('A train epoch ended.')
            self.train_epcoh_end_counter += 1

        def on_validation_epoch_end(self, trainer, model, unused=None):
            print('A val epoch ended.')
            self.validation_epoch_end_counter += 1

        def on_train_end(self, trainer, model):
            print("Training ends:"
                  f"epcoh_end_counter={self.epcoh_end_counter}, "
                  f"train_epcoh_end_counter={self.train_epcoh_end_counter}, "
                  f"validation_epoch_end_counter={self.validation_epoch_end_counter} \n")
            assert self.train_epcoh_end_counter <= epochs
            assert self.epcoh_end_counter == self.train_epcoh_end_counter + self.validation_epoch_end_counter

    callbacks = [MyDummyCallback()]

    if version.parse(torch.__version__) < version.parse('1.13'):
        """
        torch.distributed.ReduceOp is used in ModelCheckpoint and EarlyStopping.
        Since torch 1.13, it doesn't support condition check in Lightning code.
        Broken line in lightning code (https://github.com/Lightning-AI/lightning/blob/master/src/pytorch_lightning/strategies/horovod.py#L179)
        Below error will be thrown:
        >>> from torch.distributed import ReduceOp
        >>> op = None
        >>> op in (ReduceOp.SUM, None)
        Traceback (most recent call last):
            File "<stdin>", line 1, in <module>
            TypeError: __eq__(): incompatible function arguments. The following argument types are supported:
            1. (self: torch._C._distributed_c10d.ReduceOp, arg0: c10d::ReduceOp::RedOpType) -> bool
            2. (self: torch._C._distributed_c10d.ReduceOp, arg0: torch._C._distributed_c10d.ReduceOp) -> bool

        Invoked with: <torch.distributed.distributed_c10d.ReduceOp object at 0x7fba78c9e0b0>, None
        """
        # ModelCheckpoint
        from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
        callbacks.append(ModelCheckpoint(monitor='val_loss', mode="min",
                                         save_top_k=1, verbose=True))
        # EarlyStopping
        from pytorch_lightning.callbacks.early_stopping import EarlyStopping
        callbacks.append(EarlyStopping(monitor='val_loss',
                                       min_delta=0.001,
                                       patience=3,
                                       verbose=True,
                                       mode='min'))

    torch_estimator = hvd.TorchEstimator(backend=backend,
                                         store=store,
                                         model=model,
                                         input_shapes=[[-1, 1, 28, 28]],
                                         feature_cols=['features'],
                                         label_cols=['label'],
                                         batch_size=args.batch_size,
                                         epochs=args.epochs,
                                         validation=0.1,
                                         verbose=1,
                                         callbacks=callbacks,
                                         profiler="simple" if args.enable_profiler else None)

    torch_model = torch_estimator.fit(train_df).setOutputCols(['label_prob'])

    # Evaluate the model on the held-out test DataFrame
    pred_df = torch_model.transform(test_df)

    argmax = udf(lambda v: float(np.argmax(v)), returnType=T.DoubleType())
    pred_df = pred_df.withColumn('label_pred', argmax(pred_df.label_prob))
    evaluator = MulticlassClassificationEvaluator(predictionCol='label_pred', labelCol='label', metricName='accuracy')
    print('Test accuracy:', evaluator.evaluate(pred_df))

    spark.stop()


if __name__ == '__main__':
    args = parser.parse_args()
    train_model(args)
