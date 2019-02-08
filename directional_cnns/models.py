from tensorflow.python.keras.models import load_model

from directional_cnns.cloudml_utils import create_local_copy


class ModelLoader:
    """
    Handle for model allows passing around a (trained) model, without having to keep it in memory.
    It's simply referred to by its file name.
    """

    def __init__(self, file, name) -> None:
        super().__init__()
        self.file = file
        self.name = name

    def load(self):
        return load_model(create_local_copy(self.file))
