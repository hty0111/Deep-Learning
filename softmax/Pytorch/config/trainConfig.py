from .baseConfig import BaseConfig

class TrainConfig(BaseConfig):
    def initialize(self):
        parser = BaseConfig.initialize(self)
        return parser
