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


def _add_meta_graph_for_mode(  # self,
        estimator,
        builder,
        # input_receiver_fn_map,
        input_receiver_fn,
        checkpoint_path,
        # strip_default_attrs,
        save_variables=True,
        mode=model_fn_lib.ModeKeys.PREDICT,
        export_tags=None,
        check_variables=True):
    # pylint: disable=line-too-long
    """Loads variables and adds them along with a `tf.MetaGraphDef` for saving.

    Args:
      builder: instance of `tf.saved_modle.builder.SavedModelBuilder` that will
        be used for saving.
      input_receiver_fn_map: dict of `tf.estimator.ModeKeys` to
        `input_receiver_fn` mappings, where the `input_receiver_fn` is a
        function that takes no argument and returns the appropriate subclass of
        `InputReceiver`.
      checkpoint_path: The checkpoint path to export.  If `None` (the default),
        the most recent checkpoint found within the model directory is chosen.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the `NodeDef`s. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_variables: bool, whether variables should be saved. If `False`, just
        the `tf.MetaGraphDef` will be saved. Note that `save_variables` should
        only be `True` for the first call to this function, and the
        `SavedModelBuilder` will raise an error if that is not the case.
      mode: `tf.estimator.ModeKeys` value indicating which mode will be
        exported.
      export_tags: The set of tags with which to save `tf.MetaGraphDef`. If
        `None`, a default set will be selected to matched the passed mode.
      check_variables: bool, whether to check the checkpoint has all variables.

    Raises:
      ValueError: if `save_variables` is `True` and `check_variable` is `False`.
    """
    # pylint: enable=line-too-long
    if export_tags is None:
        export_tags = model_fn_lib.EXPORT_TAG_MAP[mode]
    # input_receiver_fn = input_receiver_fn_map[mode]

    with ops.Graph().as_default() as g:
        # estimator._create_and_assert_global_step(g)
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


def export_saved_model(estimator, export_dir_base, checkpoint_path, serving_input_receiver_fn):
    # pylint: enable=line-too-long
    # if not input_receiver_fn:
    #     raise ValueError('An input_receiver_fn must be defined.')

    # input_receiver_fn_map = {mode: input_receiver_fn}

    # return self._export_all_saved_models(
    #     export_dir_base,
    #     input_receiver_fn_map,
    #     assets_extra=assets_extra,
    #     as_text=as_text,
    #     checkpoint_path=checkpoint_path,
    #     strip_default_attrs=strip_default_attrs)

    # pylint: enable=line-too-long
    # TODO(b/65561022): Consider allowing multiple input_receiver_fns per mode.
    with context.graph_mode():
        # if not checkpoint_path:
        #     # Locate the latest checkpoint
        #     checkpoint_path = checkpoint_management.latest_checkpoint(
        #         self._model_dir)
        # if not checkpoint_path:
        #     raise ValueError("Couldn't find trained model at %s." % self._model_dir)

        export_dir = export_helpers.get_timestamped_export_dir(export_dir_base)
        temp_export_dir = export_helpers.get_temp_export_dir(export_dir)

        builder = saved_model_builder.SavedModelBuilder(temp_export_dir)

        save_variables = True
        # Note that the order in which we run here matters, as the first
        # mode we pass through will be used to save the variables. We run TRAIN
        # first, as that is also the mode used for checkpoints, and therefore
        # we are not likely to have vars in PREDICT that are not in the checkpoint
        # created by TRAIN.
        # if input_receiver_fn_map.get(model_fn_lib.ModeKeys.TRAIN):
        #     self._add_meta_graph_for_mode(
        #         builder, input_receiver_fn_map, checkpoint_path,
        #         strip_default_attrs, save_variables,
        #         mode=model_fn_lib.ModeKeys.TRAIN)
        #     save_variables = False
        # if input_receiver_fn_map.get(model_fn_lib.ModeKeys.EVAL):
        #     self._add_meta_graph_for_mode(
        #         builder, input_receiver_fn_map, checkpoint_path,
        #         strip_default_attrs, save_variables,
        #         mode=model_fn_lib.ModeKeys.EVAL)
        #     save_variables = False
        # if input_receiver_fn_map.get(model_fn_lib.ModeKeys.PREDICT):
        # self._add_meta_graph_for_mode(
        #     builder, input_receiver_fn_map, checkpoint_path,
        #     strip_default_attrs, save_variables,
        #     mode=model_fn_lib.ModeKeys.PREDICT)
        _add_meta_graph_for_mode(
            estimator,
            builder,
            serving_input_receiver_fn,
            checkpoint_path,
            save_variables)
        save_variables = False

        if save_variables:
            raise ValueError('No valid modes for exporting found.')

    # builder.save(as_text)

    # # Add the extra assets
    # if assets_extra:
    #   assets_extra_path = os.path.join(compat.as_bytes(temp_export_dir),
    #                                    compat.as_bytes('assets.extra'))
    #   for dest_relative, source in assets_extra.items():
    #     dest_absolute = os.path.join(compat.as_bytes(assets_extra_path),
    #                                  compat.as_bytes(dest_relative))
    #     dest_path = os.path.dirname(dest_absolute)
    #     gfile.MakeDirs(dest_path)
    #     gfile.Copy(source, dest_absolute)

    gfile.Rename(temp_export_dir, export_dir)
    return export_dir
