from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modeling
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("layers", "-1,-2,-3,-4", "")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "export_dir", None,
    "mode saved dir")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")


def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    features = {
        "unique_ids":
            tf.placeholder(shape=[None], dtype=tf.int32, name="unique_ids"),
        "input_ids":
            tf.placeholder(
                shape=[None, FLAGS.max_seq_length],
                dtype=tf.int32, name="input_ids"),
        "input_mask":
            tf.placeholder(
                shape=[None, FLAGS.max_seq_length],
                dtype=tf.int32, name="input_mask"),
        "input_type_ids":
            tf.placeholder(
                shape=[None, FLAGS.max_seq_length],
                dtype=tf.int32, name="input_type_ids"),
    }
    return tf.estimator.export.build_raw_serving_input_receiver_fn(features=features)


def model_fn_builder(bert_config, init_checkpoint, layer_indexes):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]

        model = modeling.BertModel(
            config=bert_config,
            is_training=False,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=input_type_ids)

        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))

        tvars = tf.trainable_variables()
        (assignment_map,
         initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
            tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        print(100 * "-")
        for kv in assignment_map:
            print("global" in kv, kv)
        print(100 * "-")
        for kv in initialized_variable_names:
            print(kv)
        print(100 * "+")

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        all_layers = model.get_all_encoder_layers()

        predictions = {
            "unique_id": unique_ids,
        }

        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]

        output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        return output_spec

    return model_fn


def main(_):
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    layer_indexes = [int(x) for x in FLAGS.layers.split(",")]

    model_fn = model_fn_builder(layer_indexes=layer_indexes,
                                bert_config=bert_config,
                                init_checkpoint=FLAGS.init_checkpoint)
    model_dir = None
    config = None
    params = None

    estimator = tf.estimator.Estimator(model_fn, model_dir, config, params)

    estimator.export_saved_model(export_dir_base=FLAGS.export_dir,
                                 checkpoint_path=FLAGS.init_checkpoint,
                                 serving_input_receiver_fn=serving_input_receiver_fn())


if __name__ == '__main__':
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("init_checkpoint")
    flags.mark_flag_as_required("export_dir")
    tf.app.run()
