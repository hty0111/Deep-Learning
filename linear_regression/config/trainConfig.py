from .baseConfig import BaseConfig

class TrainConfig(BaseConfig):
    def initialize(self):
        parser = BaseConfig.initialize(self)
        parser.add_argument("--true_w", type=float, nargs="+", default=[2])
        parser.add_argument("--true_b", type=float, default=4.2)
        parser.add_argument("--point_num", type=int, default=100)
        parser.add_argument("--num_workers", type=int, default=4)
        return parser
