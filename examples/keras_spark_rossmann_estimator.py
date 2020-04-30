# Copyright 2017 onwards, fast.ai, Inc.
# Modifications copyright (C) 2019 Uber Technologies, Inc.
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

import argparse
import datetime
import os
from distutils.version import LooseVersion

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark import SparkConf, Row
from pyspark.sql import SparkSession

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Embedding, Concatenate, Dense, Flatten, Reshape, BatchNormalization, Dropout

import horovod.spark.keras as hvd
from horovod.spark.common.store import Store

parser = argparse.ArgumentParser(description='Keras Spark Rossmann Estimator Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark cluster to use for training. If set to None, uses current default cluster. Cluster'
                         'should be set up to provide a Spark task per multiple CPU cores, or per GPU, e.g. by'
                         'supplying `-c <NUM_GPUS>` in Spark Standalone mode')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--learning_rate', type=float, default=0.0001,
                    help='initial learning rate')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument('--sample-rate', type=float,
                    help='desired sampling rate. Useful to set to low number (e.g. 0.01) to make sure that '
                         'end-to-end process works')
parser.add_argument('--data-dir', default='file://' + os.getcwd(),
                    help='location of data on local filesystem (prefixed with file://) or on HDFS')
parser.add_argument('--local-submission-csv', default='submission.csv',
                    help='output submission predictions CSV')
parser.add_argument('--local-checkpoint-file', default='checkpoint',
                    help='model checkpoint')
parser.add_argument('--work-dir', default='/tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')

if __name__ == '__main__':
    args = parser.parse_args()

    # ================ #
    # DATA PREPARATION #
    # ================ #

    print('================')
    print('Data preparation')
    print('================')

    # Create Spark session for data preparation.
    conf = SparkConf().setAppName('Keras Spark Rossmann Estimator Example').set('spark.sql.shuffle.partitions', '16')
    if args.master:
        conf.setMaster(args.master)
    elif args.num_proc:
        conf.setMaster('local[{}]'.format(args.num_proc))
    spark = SparkSession.builder.config(conf=conf).getOrCreate()

    train_csv = spark.read.csv('%s/train.csv' % args.data_dir, header=True)
    test_csv = spark.read.csv('%s/test.csv' % args.data_dir, header=True)

    store_csv = spark.read.csv('%s/store.csv' % args.data_dir, header=True)
    store_states_csv = spark.read.csv('%s/store_states.csv' % args.data_dir, header=True)
    state_names_csv = spark.read.csv('%s/state_names.csv' % args.data_dir, header=True)
    google_trend_csv = spark.read.csv('%s/googletrend.csv' % args.data_dir, header=True)
    weather_csv = spark.read.csv('%s/weather.csv' % args.data_dir, header=True)


    def expand_date(df):
        df = df.withColumn('Date', df.Date.cast(T.DateType()))
        return df \
            .withColumn('Year', F.year(df.Date)) \
            .withColumn('Month', F.month(df.Date)) \
            .withColumn('Week', F.weekofyear(df.Date)) \
            .withColumn('Day', F.dayofmonth(df.Date))


    def prepare_google_trend():
        # Extract week start date and state.
        google_trend_all = google_trend_csv \
            .withColumn('Date', F.regexp_extract(google_trend_csv.week, '(.*?) -', 1)) \
            .withColumn('State', F.regexp_extract(google_trend_csv.file, 'Rossmann_DE_(.*)', 1))

        # Map state NI -> HB,NI to align with other data sources.
        google_trend_all = google_trend_all \
            .withColumn('State', F.when(google_trend_all.State == 'NI', 'HB,NI').otherwise(google_trend_all.State))

        # Expand dates.
        return expand_date(google_trend_all)


    def add_elapsed(df, cols):
        def add_elapsed_column(col, asc):
            def fn(rows):
                last_store, last_date = None, None
                for r in rows:
                    if last_store != r.Store:
                        last_store = r.Store
                        last_date = r.Date
                    if r[col]:
                        last_date = r.Date
                    fields = r.asDict().copy()
                    fields[('After' if asc else 'Before') + col] = (r.Date - last_date).days
                    yield Row(**fields)
            return fn

        df = df.repartition(df.Store)
        for asc in [False, True]:
            sort_col = df.Date.asc() if asc else df.Date.desc()
            rdd = df.sortWithinPartitions(df.Store.asc(), sort_col).rdd
            for col in cols:
                rdd = rdd.mapPartitions(add_elapsed_column(col, asc))
            df = rdd.toDF()
        return df


    def prepare_df(df):
        num_rows = df.count()

        # Expand dates.
        df = expand_date(df)

        df = df \
            .withColumn('Open', df.Open != '0') \
            .withColumn('Promo', df.Promo != '0') \
            .withColumn('StateHoliday', df.StateHoliday != '0') \
            .withColumn('SchoolHoliday', df.SchoolHoliday != '0')

        # Merge in store information.
        store = store_csv.join(store_states_csv, 'Store')
        df = df.join(store, 'Store')

        # Merge in Google Trend information.
        google_trend_all = prepare_google_trend()
        df = df.join(google_trend_all, ['State', 'Year', 'Week']).select(df['*'], google_trend_all.trend)

        # Merge in Google Trend for whole Germany.
        google_trend_de = google_trend_all[google_trend_all.file == 'Rossmann_DE']
        df = df.join(google_trend_de, ['Year', 'Week']).select(df['*'], google_trend_all.trend.alias('trend_de'))

        # Merge in weather.
        weather = weather_csv.join(state_names_csv, weather_csv.file == state_names_csv.StateName)
        df = df.join(weather, ['State', 'Date'])

        # Fix null values.
        df = df \
            .withColumn('CompetitionOpenSinceYear', F.coalesce(df.CompetitionOpenSinceYear, F.lit(1900))) \
            .withColumn('CompetitionOpenSinceMonth', F.coalesce(df.CompetitionOpenSinceMonth, F.lit(1))) \
            .withColumn('Promo2SinceYear', F.coalesce(df.Promo2SinceYear, F.lit(1900))) \
            .withColumn('Promo2SinceWeek', F.coalesce(df.Promo2SinceWeek, F.lit(1)))

        # Days & months competition was open, cap to 2 years.
        df = df.withColumn('CompetitionOpenSince',
                           F.to_date(F.format_string('%s-%s-15', df.CompetitionOpenSinceYear,
                                                     df.CompetitionOpenSinceMonth)))
        df = df.withColumn('CompetitionDaysOpen',
                           F.when(df.CompetitionOpenSinceYear > 1900,
                                  F.greatest(F.lit(0), F.least(F.lit(360 * 2), F.datediff(df.Date, df.CompetitionOpenSince))))
                           .otherwise(0))
        df = df.withColumn('CompetitionMonthsOpen', (df.CompetitionDaysOpen / 30).cast(T.IntegerType()))

        # Days & weeks of promotion, cap to 25 weeks.
        df = df.withColumn('Promo2Since',
                           F.expr('date_add(format_string("%s-01-01", Promo2SinceYear), (Promo2SinceWeek - 1) * 7)'))
        df = df.withColumn('Promo2Days',
                           F.when(df.Promo2SinceYear > 1900,
                                  F.greatest(F.lit(0), F.least(F.lit(25 * 7), F.datediff(df.Date, df.Promo2Since))))
                           .otherwise(0))
        df = df.withColumn('Promo2Weeks', (df.Promo2Days / 7).cast(T.IntegerType()))

        # Check that we did not lose any rows through inner joins.
        assert num_rows == df.count(), 'lost rows in joins'
        return df


    def build_vocabulary(df, cols):
        vocab = {}
        for col in cols:
            values = [r[0] for r in df.select(col).distinct().collect()]
            col_type = type([x for x in values if x is not None][0])
            default_value = col_type()
            vocab[col] = sorted(values, key=lambda x: x or default_value)
        return vocab


    def cast_columns(df, cols):
        for col in cols:
            df = df.withColumn(col, F.coalesce(df[col].cast(T.FloatType()), F.lit(0.0)))
        return df


    def lookup_columns(df, vocab):
        def lookup(mapping):
            def fn(v):
                return mapping.index(v)
            return F.udf(fn, returnType=T.IntegerType())

        for col, mapping in vocab.items():
            df = df.withColumn(col, lookup(mapping)(df[col]))
        return df


    if args.sample_rate:
        train_csv = train_csv.sample(withReplacement=False, fraction=args.sample_rate)
        test_csv = test_csv.sample(withReplacement=False, fraction=args.sample_rate)

    # Prepare data frames from CSV files.
    train_df = prepare_df(train_csv).cache()
    test_df = prepare_df(test_csv).cache()

    # Add elapsed times from holidays & promos, the data spanning training & test datasets.
    elapsed_cols = ['Promo', 'StateHoliday', 'SchoolHoliday']
    elapsed = add_elapsed(train_df.select('Date', 'Store', *elapsed_cols)
                          .unionAll(test_df.select('Date', 'Store', *elapsed_cols)),
                          elapsed_cols)

    # Join with elapsed times.
    train_df = train_df \
        .join(elapsed, ['Date', 'Store']) \
        .select(train_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])
    test_df = test_df \
        .join(elapsed, ['Date', 'Store']) \
        .select(test_df['*'], *[prefix + col for prefix in ['Before', 'After'] for col in elapsed_cols])

    # Filter out zero sales.
    train_df = train_df.filter(train_df.Sales > 0)

    print('===================')
    print('Prepared data frame')
    print('===================')
    train_df.show()

    categorical_cols = [
        'Store', 'State', 'DayOfWeek', 'Year', 'Month', 'Day', 'Week', 'CompetitionMonthsOpen', 'Promo2Weeks', 'StoreType',
        'Assortment', 'PromoInterval', 'CompetitionOpenSinceYear', 'Promo2SinceYear', 'Events', 'Promo',
        'StateHoliday', 'SchoolHoliday'
    ]

    continuous_cols = [
        'CompetitionDistance', 'Max_TemperatureC', 'Mean_TemperatureC', 'Min_TemperatureC', 'Max_Humidity',
        'Mean_Humidity', 'Min_Humidity', 'Max_Wind_SpeedKm_h', 'Mean_Wind_SpeedKm_h', 'CloudCover', 'trend', 'trend_DE',
        'BeforePromo', 'AfterPromo', 'AfterStateHoliday', 'BeforeStateHoliday', 'BeforeSchoolHoliday', 'AfterSchoolHoliday'
    ]

    all_cols = categorical_cols + continuous_cols

    # Select features.
    train_df = train_df.select(*(all_cols + ['Sales', 'Date'])).cache()
    test_df = test_df.select(*(all_cols + ['Id', 'Date'])).cache()

    # Build vocabulary of categorical columns.
    vocab = build_vocabulary(train_df.select(*categorical_cols)
                             .unionAll(test_df.select(*categorical_cols)).cache(),
                             categorical_cols)

    # Cast continuous columns to float & lookup categorical columns.
    train_df = cast_columns(train_df, continuous_cols + ['Sales'])
    train_df = lookup_columns(train_df, vocab)
    test_df = cast_columns(test_df, continuous_cols)
    test_df = lookup_columns(test_df, vocab)

    # Split into training & validation.
    # Test set is in 2015, use the same period in 2014 from the training set as a validation set.
    test_min_date = test_df.agg(F.min(test_df.Date)).collect()[0][0]
    test_max_date = test_df.agg(F.max(test_df.Date)).collect()[0][0]
    one_year = datetime.timedelta(365)
    train_df = train_df.withColumn('Validation',
                                   (train_df.Date > test_min_date - one_year) & (train_df.Date <= test_max_date - one_year))

    # Determine max Sales number.
    max_sales = train_df.agg(F.max(train_df.Sales)).collect()[0][0]

    # Convert Sales to log domain
    train_df = train_df.withColumn('Sales', F.log(train_df.Sales))

    print('===================================')
    print('Data frame with transformed columns')
    print('===================================')
    train_df.show()

    print('================')
    print('Data frame sizes')
    print('================')
    train_rows = train_df.filter(~train_df.Validation).count()
    val_rows = train_df.filter(train_df.Validation).count()
    test_rows = test_df.count()
    print('Training: %d' % train_rows)
    print('Validation: %d' % val_rows)
    print('Test: %d' % test_rows)

    # ============== #
    # MODEL TRAINING #
    # ============== #

    print('==============')
    print('Model training')
    print('==============')


    def exp_rmspe(y_true, y_pred):
        """Competition evaluation metric, expects logarithic inputs."""
        pct = tf.square((tf.exp(y_true) - tf.exp(y_pred)) / tf.exp(y_true))
        # Compute mean excluding stores with zero denominator.
        x = tf.reduce_sum(tf.where(y_true > 0.001, pct, tf.zeros_like(pct)))
        y = tf.reduce_sum(tf.where(y_true > 0.001, tf.ones_like(pct), tf.zeros_like(pct)))
        return tf.sqrt(x / y)


    def act_sigmoid_scaled(x):
        """Sigmoid scaled to logarithm of maximum sales scaled by 20%."""
        return tf.nn.sigmoid(x) * tf.math.log(max_sales) * 1.2


    CUSTOM_OBJECTS = {'exp_rmspe': exp_rmspe,
                      'act_sigmoid_scaled': act_sigmoid_scaled}

    # Disable GPUs when building the model to prevent memory leaks
    if LooseVersion(tf.__version__) >= LooseVersion('2.0.0'):
        # See https://github.com/tensorflow/tensorflow/issues/33168
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        K.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

    # Build the model.
    inputs = {col: Input(shape=(1,), name=col) for col in all_cols}
    embeddings = [Embedding(len(vocab[col]), 10, input_length=1, name='emb_' + col)(inputs[col])
                  for col in categorical_cols]
    continuous_bn = Concatenate()([Reshape((1, 1), name='reshape_' + col)(inputs[col])
                                   for col in continuous_cols])
    continuous_bn = BatchNormalization()(continuous_bn)
    x = Concatenate()(embeddings + [continuous_bn])
    x = Flatten()(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dense(500, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.00005))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation=act_sigmoid_scaled)(x)
    model = tf.keras.Model([inputs[f] for f in all_cols], output)
    model.summary()

    opt = tf.keras.optimizers.Adam(lr=args.learning_rate, epsilon=1e-3)

    # Horovod: run training.
    store = Store.create(args.work_dir)
    keras_estimator = hvd.KerasEstimator(num_proc=args.num_proc,
                                         store=store,
                                         model=model,
                                         optimizer=opt,
                                         loss='mae',
                                         metrics=[exp_rmspe],
                                         custom_objects=CUSTOM_OBJECTS,
                                         feature_cols=all_cols,
                                         label_cols=['Sales'],
                                         validation='Validation',
                                         batch_size=args.batch_size,
                                         epochs=args.epochs,
                                         verbose=2)

    keras_model = keras_estimator.fit(train_df).setOutputCols(['Sales'])

    history = keras_model.getHistory()
    best_val_rmspe = min(history['val_exp_rmspe'])
    print('Best RMSPE: %f' % best_val_rmspe)

    # Save the trained model.
    keras_model.save(args.local_checkpoint_file)
    print('Written checkpoint to %s' % args.local_checkpoint_file)

    # ================ #
    # FINAL PREDICTION #
    # ================ #

    print('================')
    print('Final prediction')
    print('================')

    pred_df = keras_model.transform(test_df)
    # Convert from log domain to real Sales numbers
    pred_df = pred_df.withColumn('Sales', F.exp(pred_df.Sales))
    submission_df = pred_df.select(pred_df.Id.cast(T.IntegerType()), pred_df.Sales).toPandas()
    submission_df.sort_values(by=['Id']).to_csv(args.local_submission_csv, index=False)
    print('Saved predictions to %s' % args.local_submission_csv)

    spark.stop()
