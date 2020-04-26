from tensorflow import estimator
from tensorflow import keras
# ----------------------------------------
from .base import Processor
# --------------------------------------------------


class TrainProcessor(Processor):

    def __init__(self, config):
        super().__init__(config)

        # ----------------------------------------
        self._estimator = estimator.Estimator(
            model_fn=self._model_fn,
            config=self._config_fn,
            params={'fakeParam': 1},
            warm_start_from=None
        )

    def __call__(self):
        self._estimator.train(
            input_fn=lambda: self._input_fn()
        )

    def _model_fn(self):
        pass

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


