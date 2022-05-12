from matplotlib import pyplot as plt
import os

class Visualize():
    def __init__(self, image_root) -> None:
        self.image_root = image_root

    def plot_train(self, loss, acc):
        plt.plot(loss, label="loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.image_root, "loss.png"))
        plt.close()
        plt.plot(acc, label="acc")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.image_root, "accuracy.png"))
        plt.close()
