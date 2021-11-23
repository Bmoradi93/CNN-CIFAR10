import tensorflow as tf
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import itertools
from modules import cifar10_cnn

if __name__ == "__main__":
    # Creating an object from cifar10_cnn
    cnn_object = cifar10_cnn()

    # Prepare cifar10 dataset
    cnn_object.get_dataset()

    # Building the CNN model
    cnn_object.build_cnn_model()

    # Training the CNN model
    cnn_object.train_cnn_model()

    # Evaluate the CNN model
    cnn_object.evaluate_cnn_model()

    # Prediction
    cnn_object.predict()

    # Plot Results
    cnn_object.plot_results()