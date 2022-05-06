import argparse
import torch

class BaseConfig():
    def __init__(self) -> None:
        self.initialized = False

    def getArgs(self):
        if self.initialized == False:
            parser = self.initialize()   
        args = parser.parse_args()
        self.printArgs(args)
        self.setGPU(args)
        return args
    
    def initialize(self):
        parser = argparse.ArgumentParser(description="Basic arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument("--data_root",  type=str,   default="./datasets", help="Root of datasets")
        parser.add_argument("--model_root", type=str,   default="./checkpoints", help="Root of trained models.")
        parser.add_argument("--batch_size", type=int,   default=4)
        parser.add_argument("--epochs",     type=int,   default=100)
        parser.add_argument("--lr",         type=float, default=0.01)
        parser.add_argument("--gpu",        type=str,   default="0")
        self.initialized = True
        return parser

    def printArgs(self, args):
        message = ''
        message += '----------------- Configs ---------------\n'
        for arg in vars(args):  # vars() 函数返回对象object的属性和属性值的字典对象
            message += "{:>20}: {:<20}\n".format(str(arg), str(getattr(args, arg)))
        message += '----------------- End -------------------'
        print(message)

    def setGPU(self, args):
        str_ids = args.gpu.split(',')
        gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                gpu_ids.append(id)
        if len(gpu_ids) > 0:
            torch.cuda.set_device(gpu_ids[0])