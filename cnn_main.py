import tensorflow as tf
from tensorflow import data
from datetime import datetime
import multiprocessing
import shutil
import os
tf.reset_default_graph()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
MODEL_NAME = 'sms-class-model-01'
TRAIN_DATA_FILES_PATTERN = 'data/sms-spam/train-*.tsv'
VALID_DATA_FILES_PATTERN = 'data/sms-spam/valid-*.tsv'

VOCAB_LIST_FILE = 'data/sms-spam/vocab_list.tsv'
N_WORDS_FILE = 'data/sms-spam/n_words.tsv'
RESUME_TRAINING = False
MULTI_TRAINING = True

MAX_DOCUMENT_LENGTH = 100
PAD_WORD = '#=KS=#'
WEIGHT_COLUNM_NAME = 'weight'
HEADER = ['class', 'instances']
HEADER_DEFAULTS = [['NA'], ['NA']]

TEXT_FEATURE_NAME = 'instances'

TARGET_NAME = 'class'
WEIGHT_COLUMN_NAME = 'weight'
TARGET_LABELS = ['spam', 'ham']

TRAIN_SIZE = 4179
NUM_EPOCHS = 10
BATCH_SIZE = 250
EVAL_AFTER_SEC = 60
TOTAL_STEPS = int((TRAIN_SIZE/BATCH_SIZE) * NUM_EPOCHS)

model_dir = 'trained_models/{}'.format(MODEL_NAME)

run_config = tf.estimator.RunConfig(log_step_count_steps=5000,
                                    tf_random_seed=19830610,
                                    model_dir=model_dir)

with open(N_WORDS_FILE) as file:
    N_WORDS = int(file.read())+2


MULTI_THREADING = True


def parse_tsv_row(tsv_row):
    columns = tf.decode_csv(tsv_row, record_defaults=HEADER_DEFAULTS, field_delim='\t')
    features = dict(zip(HEADER, columns))

    target = features.pop(TARGET_NAME)

    # giving more weight to "spam" records are the are only 13% of the training set
    features[WEIGHT_COLUNM_NAME] = tf.cond(tf.equal(target, 'spam'), lambda: 6.6, lambda: 1.0)

    return features, target


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(TARGET_LABELS))
    return table.lookup(label_string_tensor)


def input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
             skip_header_lines=0,
             num_epochs=1,
             batch_size=200):
    shuffle = True if mode == tf.estimator.ModeKeys.TRAIN else False

    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1

    buffer_size = 2 * batch_size + 1

    print("")
    print("* data input_fn:")
    print("================")
    print("Input file(s): {}".format(files_name_pattern))
    print("Batch size: {}".format(batch_size))
    print("Epoch Count: {}".format(num_epochs))
    print("Mode: {}".format(mode))
    print("Thread Count: {}".format(num_threads))
    print("Shuffle: {}".format(shuffle))
    print("================")
    print("")

    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)

    dataset = dataset.skip(skip_header_lines)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.map(lambda tsv_row: parse_tsv_row(tsv_row),
                          num_parallel_calls=num_threads)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.prefetch(buffer_size)

    iterator = dataset.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, parse_label_column(target)

def process_text(text_feature):
    # Load vocabolary lookup table to map word => word_id
    vocab_table = tf.contrib.lookup.index_table_from_file(vocabulary_file=VOCAB_LIST_FILE,
                                                          num_oov_buckets=1, default_value=-1)
    # Get text feature
    smss = text_feature
    # Split text to words -> this will produce sparse tensor with variable-lengthes (word count) entries
    words = tf.string_split(smss)
    # Convert sparse tensor to dense tensor by padding each entry to match the longest in the batch
    dense_words = tf.sparse_tensor_to_dense(words, default_value=PAD_WORD)
    # Convert word to word_ids via the vocab lookup table
    word_ids = vocab_table.lookup(dense_words)
    # Create a word_ids padding
    padding = tf.constant([[0, 0], [0, MAX_DOCUMENT_LENGTH]])
    # Pad all the word_ids entries to the maximum document length
    word_ids_padded = tf.pad(word_ids, padding)
    word_id_vector = tf.slice(word_ids_padded, [0, 0], [-1, MAX_DOCUMENT_LENGTH])

    # Return the final word_id_vector
    return word_id_vector


def model_fn(features, labels, mode, params):
    hidden_units = params.hidden_units
    output_layer_size = len(TARGET_LABELS)
    embedding_size = params.embedding_size
    window_size = params.window_size
    stride = int(window_size / 2)
    filters = params.filters

    # word_id_vector
    word_id_vector = process_text(features[TEXT_FEATURE_NAME])
    # print("word_id_vector: {}".format(word_id_vector)) # (?, MAX_DOCUMENT_LENGTH)

    # layer to take each word_id and convert it into vector (embeddings)
    word_embeddings = tf.contrib.layers.embed_sequence(word_id_vector, vocab_size=N_WORDS,
                                                       embed_dim=embedding_size)
    # print("word_embeddings: {}".format(word_embeddings)) # (?, MAX_DOCUMENT_LENGTH, embbeding_size)

    # convolution
    words_conv = tf.layers.conv1d(word_embeddings, filters=filters, kernel_size=window_size,
                                  strides=stride, padding='SAME', activation=tf.nn.relu)

    # print("words_conv: {}".format(words_conv)) # (?, MAX_DOCUMENT_LENGTH/stride, filters)

    words_conv_shape = words_conv.get_shape()
    dim = words_conv_shape[1] * words_conv_shape[2]
    input_layer = tf.reshape(words_conv, [-1, dim])
    # print("input_layer: {}".format(input_layer)) # (?, (MAX_DOCUMENT_LENGTH/stride)*filters)

    if hidden_units is not None:

        # Create a fully-connected layer-stack based on the hidden_units in the params
        hidden_layers = tf.contrib.layers.stack(inputs=input_layer,
                                                layer=tf.contrib.layers.fully_connected,
                                                stack_args=hidden_units,
                                                activation_fn=tf.nn.relu)
        # print("hidden_layers: {}".format(hidden_layers)) # (?, last-hidden-layer-size)

    else:
        hidden_layers = input_layer

    # Connect the output layer (logits) to the hidden layer (no activation fn)
    logits = tf.layers.dense(inputs=hidden_layers,
                             units=output_layer_size,
                             activation=None)

    # print("logits: {}".format(logits)) # (?, output_layer_size)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Convert predicted_indices back into strings
        predictions = {
            'class': tf.gather(TARGET_LABELS, predicted_indices),
            'probabilities': probabilities
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }

        # Provide an estimator spec for `ModeKeys.PREDICT` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions,
                                          export_outputs=export_outputs)

    # weights
    weights = features[WEIGHT_COLUNM_NAME]

    # Calculate loss using softmax cross entropy
    loss = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels,
        weights=weights
    )

    tf.summary.scalar('loss', loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Create Optimiser
        optimizer = tf.train.AdamOptimizer(params.learning_rate)

        # Create training operation
        train_op = optimizer.minimize(
            loss=loss, global_step=tf.train.get_global_step())

        # Provide an estimator spec for `ModeKeys.TRAIN` modes.
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)

    if mode == tf.estimator.ModeKeys.EVAL:
        probabilities = tf.nn.softmax(logits)
        predicted_indices = tf.argmax(probabilities, 1)

        # Return accuracy and area under ROC curve metrics
        labels_one_hot = tf.one_hot(
            labels,
            depth=len(TARGET_LABELS),
            on_value=True,
            off_value=False,
            dtype=tf.bool
        )

        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(labels, predicted_indices, weights=weights),
            'auroc': tf.metrics.auc(labels_one_hot, probabilities, weights=weights)
        }

        # Provide an estimator spec for `ModeKeys.EVAL` modes.
        return tf.estimator.EstimatorSpec(mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)


def create_estimator(run_config, hparams):
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=hparams,
                                       config=run_config)

    print("")
    print("Estimator Type: {}".format(type(estimator)))
    print("")

    return estimator

hparams = tf.contrib.training.HParams(num_epochs=NUM_EPOCHS,
                                      batch_size=BATCH_SIZE,
                                      embedding_size=3,
                                      window_size=3,
                                      filters=2,
                                      hidden_units=None,
                                      max_steps=TOTAL_STEPS,
                                      learning_rate=0.01)


def serving_input_fn():
    receiver_tensor = {
        'instances': tf.placeholder(tf.string, [None]),
    }

    features = {
        key: tensor
        for key, tensor in receiver_tensor.items()
    }

    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensor)

if __name__ == '__main__':
    # ==============预测阶段=====================
    export_dir = model_dir + "/export/predict/"

    saved_model_dir = export_dir + "/" + os.listdir(path=export_dir)[-1]

    print(saved_model_dir)
    print("")

    predictor_fn = tf.contrib.predictor.from_saved_model(
        export_dir=saved_model_dir,
        signature_def_key="prediction"
    )

    output = predictor_fn(
        {
            'instances': [
                'ok, I will be with you in 5 min. see you then',
                'win 1000 cash free of charge promo hot deal sexy',
                'hot girls sexy tonight call girls waiting call chat'
            ]

        }
    )
    print(output)

    # ==============验证阶段======================
    # TRAIN_SIZE = 4179
    # TEST_SIZE = 1393
    #
    # train_input_fn = lambda: input_fn(files_name_pattern=TRAIN_DATA_FILES_PATTERN,
    #                                   mode=tf.estimator.ModeKeys.EVAL,
    #                                   batch_size=TRAIN_SIZE)
    #
    # test_input_fn = lambda: input_fn(files_name_pattern=VALID_DATA_FILES_PATTERN,
    #                                  mode=tf.estimator.ModeKeys.EVAL,
    #                                  batch_size=TEST_SIZE)
    #
    # estimator = create_estimator(run_config, hparams)
    #
    # train_results = estimator.evaluate(input_fn=train_input_fn, steps=1)
    # print()
    # print("######################################################################################")
    # print("# Train Measures: {}".format(train_results))
    # print("######################################################################################")
    #
    # test_results = estimator.evaluate(input_fn=test_input_fn, steps=1)
    # print()
    # print("######################################################################################")
    # print("# Test Measures: {}".format(test_results))
    # print("######################################################################################")


    # =============训练阶段=======================
    # train_spec = tf.estimator.TrainSpec(
    #     input_fn=lambda: input_fn(
    #         TRAIN_DATA_FILES_PATTERN,
    #         mode=tf.estimator.ModeKeys.TRAIN,
    #         num_epochs=hparams.num_epochs,
    #         batch_size=hparams.batch_size
    #     ),
    #     max_steps=hparams.max_steps,
    #     hooks=None
    # )
    #
    # eval_spec = tf.estimator.EvalSpec(
    #     input_fn=lambda: input_fn(
    #         VALID_DATA_FILES_PATTERN,
    #         mode=tf.estimator.ModeKeys.EVAL,
    #         batch_size=hparams.batch_size
    #     ),
    #     exporters=[tf.estimator.LatestExporter(
    #         name="predict",  # the name of the folder in which the model will be exported to under export
    #         serving_input_receiver_fn=serving_input_fn,
    #         exports_to_keep=1,
    #         as_text=True)],
    #     steps=None,
    #     throttle_secs=EVAL_AFTER_SEC
    # )
    # if not RESUME_TRAINING:
    #     print("Removing previous artifacts...")
    #     shutil.rmtree(model_dir, ignore_errors=True)
    # else:
    #     print("Resuming training...")
    #
    # tf.logging.set_verbosity(tf.logging.INFO)
    #
    # time_start = datetime.utcnow()
    # print("Experiment started at {}".format(time_start.strftime("%H:%M:%S")))
    #
    # estimator = create_estimator(run_config, hparams)
    #
    # tf.estimator.train_and_evaluate(estimator=estimator,
    #                                 train_spec=train_spec,
    #                                 eval_spec=eval_spec)
    # time_end = datetime.utcnow()
    # print(".......................................")
    # print("Experiment finished at {}".format(time_end.strftime("%H:%M:%S")))
    # print("")
    # time_elapsed = time_end - time_start
    # print("Experiment elapsed time: {} seconds".format(time_elapsed.total_seconds()))
