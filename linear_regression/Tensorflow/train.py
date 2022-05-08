import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from tensorflow import keras
from config.trainConfig import TrainConfig
from data.dataloader import LinearDataset
from utils.visualize import Visualize
import numpy as np
from matplotlib import pyplot as plt


def main():
    # ###########
    # config  
    # ###########  
    cfg = TrainConfig().getArgs()

    # ###########
    # load data  
    # ###########  
    x, y, dataset = LinearDataset(cfg).generate_dataset()
    # print(next(iter(dataset)))
    
    # ###########
    # model
    # ########### 
    initializer = tf.initializers.RandomNormal(stddev=0.01)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.models.Sequential([tf.keras.layers.Dense(1, kernel_initializer=initializer)])
    criterion = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.SGD(learning_rate=cfg.lr)

    # ###########
    # train
    # ###########
    with tf.device(f"/GPU:{int(cfg.gpus[0])}"):
        for epoch in range(cfg.epochs):
            for feature, label in dataset:
                    with tf.GradientTape() as tape:
                        loss = criterion(label, model(feature, training=True))
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    # print(feature.device)
                    # print(label.shape, predict.shape)
            if (epoch+1) % 5 == 0:
                print(f"epoch: {epoch+1}, loss: {loss:.4f}")

    # ###########
    # save
    # ########### 
    save_path = os.path.join(cfg.model_root, f"linear_{epoch+1}epochs.h5")
    model.save(save_path)

    # ###########
    # visualize
    # ########### 
    y_hat = model(x)
    plt.scatter(x[:, 0], y, label="True")
    plt.scatter(x[:, 0], y_hat, label="Predict")
    plt.legend()
    plt.savefig(os.path.join(cfg.image_root, "prediction.png"))


if __name__=="__main__":
    main()



