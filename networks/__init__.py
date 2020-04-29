from tensorflow import keras
# --------------------------------------------------


class NetworkFactory:

    _DEFAULT_NETWORKS = {
        'ResNet50': keras.applications.ResNet50,
        'ResNet50V2': keras.applications.ResNet50V2,
    }

    @classmethod
    def register(cls):
        pass

    @classmethod
    def get(cls, network_name, input_shape):
        network_fn = cls._DEFAULT_NETWORKS.get(network_name)
        if not network_fn:
            raise ValueError(f'Deosn’t exist network: {network_name}')
        return network_fn(
            include_top=False,
            input_tensor=None,
            input_shape=input_shape,
            pooling='avg',
        )