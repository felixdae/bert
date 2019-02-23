from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session as tf_session
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.estimator import model_fn as model_fn_lib
from tensorflow.python.framework import random_seed
from tensorflow.python.estimator.export import export as export_helpers
from tensorflow.python.training import monitored_session
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver
from tensorflow.python.framework import errors
from tensorflow.python.saved_model import builder as saved_model_builder


def _add_meta_graph_for_mode(
        estimator,
        builder,
        input_receiver_fn,
        checkpoint_path,
        save_variables=True,
        mode=model_fn_lib.ModeKeys.PREDICT,
        export_tags=None,
        check_variables=True):
    if export_tags is None:
        export_tags = model_fn_lib.EXPORT_TAG_MAP[mode]

    with ops.Graph().as_default() as g:
        random_seed.set_random_seed(estimator._config.tf_random_seed)

        input_receiver = input_receiver_fn()

        # Call the model_fn and collect the export_outputs.
        estimator_spec = estimator._call_model_fn(
            features=input_receiver.features,
            labels=getattr(input_receiver, 'labels', None),
            mode=mode,
            config=estimator.config)

        export_outputs = model_fn_lib.export_outputs_for_mode(
            mode=estimator_spec.mode,
            serving_export_outputs=estimator_spec.export_outputs,
            predictions=estimator_spec.predictions,
            loss=estimator_spec.loss,
            metrics=estimator_spec.eval_metric_ops)

        # Build the SignatureDefs from receivers and all outputs
        signature_def_map = export_helpers.build_all_signature_defs(
            input_receiver.receiver_tensors,
            export_outputs,
            getattr(input_receiver, 'receiver_tensors_alternatives', None),
            serving_only=(mode == model_fn_lib.ModeKeys.PREDICT))

        with tf_session.Session(config=estimator._session_config) as session:

            if estimator_spec.scaffold.local_init_op is not None:
                local_init_op = estimator_spec.scaffold.local_init_op
            else:
                local_init_op = monitored_session.Scaffold.default_local_init_op()

            # This saver will be used both for restoring variables now,
            # and in saving out the metagraph below. This ensures that any
            # Custom Savers stored with the Scaffold are passed through to the
            # SavedModel for restore later.
            graph_saver = estimator_spec.scaffold.saver or saver.Saver(sharded=True)

            if save_variables and not check_variables:
                raise ValueError('If `save_variables` is `True, `check_variables`'
                                 'must not be `False`.')
            if check_variables:
                try:
                    graph_saver.restore(session, checkpoint_path)
                except errors.NotFoundError as e:
                    msg = ('Could not load all requested variables from checkpoint. '
                           'Please make sure your model_fn does not expect variables '
                           'that were not saved in the checkpoint.\n\n'
                           'Encountered error with mode `{}` while restoring '
                           'checkpoint from: `{}`. Full Traceback:\n\n{}').format(
                        mode, checkpoint_path, e)
                    raise ValueError(msg)

            # We add the train op explicitly for now, so that we don't have to
            # change the Builder public interface. Note that this is a no-op
            # for prediction, where train_op is None.
            builder._add_train_op(estimator_spec.train_op)  # pylint: disable=protected-access

            meta_graph_kwargs = dict(
                tags=export_tags,
                signature_def_map=signature_def_map,
                assets_collection=ops.get_collection(
                    ops.GraphKeys.ASSET_FILEPATHS),
                strip_default_attrs=True,
                legacy_init_op=local_init_op,
                saver=graph_saver)

            if save_variables:
                builder.add_meta_graph_and_variables(
                    session, **meta_graph_kwargs)
            else:
                builder.add_meta_graph(**meta_graph_kwargs)


def export_saved_model(estimator,
                       export_dir_base,
                       checkpoint_path,
                       serving_input_receiver_fn,
                       as_text=False):
    with context.graph_mode():
        export_dir = export_helpers.get_timestamped_export_dir(export_dir_base)
        temp_export_dir = export_helpers.get_temp_export_dir(export_dir)

        builder = saved_model_builder.SavedModelBuilder(temp_export_dir)

        save_variables = True
        _add_meta_graph_for_mode(
            estimator,
            builder,
            serving_input_receiver_fn,
            checkpoint_path,
            save_variables)
        save_variables = False

        builder.save(as_text)
        if save_variables:
            raise ValueError('No valid modes for exporting found.')

    gfile.Rename(temp_export_dir, export_dir)
    return export_dir
