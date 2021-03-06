# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:58:34 2021

@author: Mertcan
"""
import numpy as np
from tensorflow.keras.models import load_model
from matplotlib import pyplot

def save_plot(examples, n):
    examples = (examples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, i+1)
        pyplot.axis("off")
        pyplot.imshow(examples[i])
    filename = "fake.png"
    pyplot.savefig(filename)
    pyplot.close()

model = load_model("C:/Users/Mertcan/Desktop/staj/g_model.h5")
n_samples = 25     ## n should always be a square of an integer.
latent_dim = 128
latent_points = np.random.normal(size=(n_samples, latent_dim))
examples = model.predict(latent_points)
save_plot(examples, int(np.sqrt(n_samples)))