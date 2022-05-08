import tensorflow as tf
from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt


class LinearDataset():
    def __init__(self, args):
        self.w = args.true_w
        self.b = args.true_b
        self.num = args.point_num
        self.batch_size = args.batch_size

    def __len__(self):
        return self.num

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]

    def generate_dataset(self):
        w = tf.constant(self.w, dtype=tf.float32)
        b = tf.constant(self.b)
        x = tf.zeros((self.num, len(w)), dtype=tf.float32)
        x += tf.random.normal(shape=x.shape)
        y = tf.matmul(x, tf.reshape(w, (-1, 1))) + b
        y += tf.random.normal(shape=y.shape)
        y = tf.reshape(y, (-1, 1))
        dataset = tf.data.Dataset.from_tensor_slices((x, y))
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(self.batch_size)
        return x, y, dataset

if __name__=="__main__":
    dataset = LinearDataset().generate_dataset()
