from matplotlib import pyplot as plt
import os

class Visualize():
    def __init__(self, cfg) -> None:
        self.image_root = cfg.image_root
        self.model_name = cfg.model_name
        self.epochs = cfg.epochs

    def plot_train(self, loss, acc):
        plt.plot(loss, label="loss")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.image_root, f"{self.model_name}_{self.epochs}epochs_loss.png"))
        plt.close()
        plt.plot(acc, label="acc")
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(self.image_root, f"{self.model_name}_{self.epochs}epochs_accuracy.png"))
        plt.close()
