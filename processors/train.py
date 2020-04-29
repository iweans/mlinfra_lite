from tensorflow import estimator
from tensorflow import keras
# ----------------------------------------
from networks import NetworkFactory
# ----------------------------------------
from .base import Processor
# --------------------------------------------------


class TrainProcessor(Processor):

    def __init__(self, config):
        super().__init__(config)
        # ----------------------------------------
        self._model_name = 'ResNet50'
        self._iter_num = 10
        self._train_labels_path = ''
        self._validate_labels_path = ''
        self._test_labels_path = ''
        # ----------------------------------------
        self._estimator = estimator.Estimator(
            model_fn=self._model_fn,
            config=self._config_fn,
            params={'fakeParam': 1},
            warm_start_from=None
        )

    def __call__(self):
        train_spec = estimator.TrainSpec(input_fn=lambda: self._input_fn(),
                                         max_steps=self._iter_num)
        eval_spec = estimator.EvalSpec(input_fn=lambda: self._input_fn(), )
        return estimator.train_and_evaluate(self._estimator,
                                            train_spec=train_spec, eval_spec=eval_spec)

    def _model_fn(self, features, labels, mode, params=None):
        network = NetworkFactory.get(self._model_name)
        # ----------------------------------------

        # ----------------------------------------

        return estimator.EstimatorSpec(
            mode=mode,

        )


    @property
    def _config_fn(self):
        return estimator.RunConfig(
            model_dir='./ckpt',
            save_checkpoints_steps=500,
            keep_checkpoint_max=10,
        )

    def _input_fn(self, data, model, num_parallel_calls=8):
        return data.create_dataset(
            callback_fn=model.create_dataset,
            num_parallel_calls=num_parallel_calls
        )

