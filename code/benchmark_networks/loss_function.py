import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math

################################# loss functions ##########################################################

def denoise_loss_mse(denoise, clean):      
  loss = tf.losses.mean_squared_error(denoise, clean)
  return tf.reduce_mean(loss)

def denoise_loss_rmse(denoise, clean):      #tmse
  loss = tf.losses.mean_squared_error(denoise, clean)
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return tf.math.sqrt(tf.reduce_mean(loss))

def denoise_loss_rrmset(denoise, clean):      #tmse
  rmse1 = denoise_loss_rmse(denoise, clean)
  rmse2 = denoise_loss_rmse(clean, tf.zeros(clean.shape[0], tf.float64))
  #loss2 = tf.losses.mean_squared_error(noise, clean)
  return rmse1/rmse2

def denoise_loss_rrmsepsd(denoise, clean):
  denoise, clean = tf.squeeze(denoise), tf.squeeze(clean)
  result = []
  for len in range(denoise.shape[0]):
    psd1,_ = plt.psd(denoise[len, :], Fs = 256)
    psd2,_ = plt.psd(clean[len, :], Fs = 256)
    psd1, psd2 = tf.convert_to_tensor(psd1,dtype=tf.float32), tf.convert_to_tensor(psd2, dtype=tf.float32)
    rmse1 = denoise_loss_rmse(psd1, psd2)
    rmse2 = denoise_loss_rmse(psd2, tf.zeros(psd2.shape))
    result.append(rmse1/ rmse2)
  return np.mean(result)

def average_correlation_coefficient(denoise, clean):
  denoise, clean = tf.squeeze(denoise), tf.squeeze(clean)
  result = []
  for len in range(denoise.shape[0]):
    temp1 = pd.Series(denoise[len, :])
    temp2 = pd.Series(clean[len, :])
    covar = temp1.cov(temp2)
    var_prod = math.sqrt(temp1.var() * temp2.var())
    result.append(covar / var_prod)  
  return np.mean(result) 