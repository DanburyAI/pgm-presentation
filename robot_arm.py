import os
import sys
sys.path.append('./deps/edward')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
import edward as ed
from edward.models import Categorical, Mixture, Normal
from collections import deque
from sklearn.model_selection import train_test_split
from scipy import stats
import seaborn as sns

ed.set_seed(42)

ARM_LENGTH_1 = 3
ARM_LENGTH_2 = 2

def get_effector_pos(theta1, theta2, degrees=False, noise=False):
    angle1 = theta1
    angle2 = theta2
    if(degrees):
        angle1 = degrees_2_rads(angle1)
        angle2 = degrees_2_rads(angle2)
    x = ARM_LENGTH_1 * np.cos(angle1) + \
        ARM_LENGTH_2 * np.cos(angle1 + angle2)
    y = ARM_LENGTH_1 * np.sin(angle1) + \
        ARM_LENGTH_2 * np.sin(angle1 + angle2)
    if noise:
        x += np.random.uniform(-0.01, 0.01, 1)
        y += np.random.uniform(-0.01, 0.01, 1)
    return x,y

def degrees_2_rads(deg):
    return deg * (np.pi / 180.)

def rads_2_degrees(rad):
    return rad * (180 / np.pi)

def scanl(f, q, ls):
    ret = [q]
    if len(ls) == 0:
        return ret
    l = deque(ls)
    x, xs = l.popleft(),l
    return ret + scanl(f, f(q,x), xs)

def gen_training(N):
    theta1 = np.random.uniform(0.0, 2*np.pi, N)
    theta2 = np.random.uniform(0.0, 2*np.pi, N)
    theta1,theta2 = np.meshgrid(theta1,theta2)
    r_x_data = np.random.normal(size=N)  # random noise
    x = ARM_LENGTH_1 * np.cos(theta1) + \
        ARM_LENGTH_2 * np.cos(theta1 + theta2) + r_x_data
    x = x.reshape(-1,1)
    r_y_data = np.random.normal(size=N)  # random noise
    y = ARM_LENGTH_1 * np.sin(theta1) + \
        ARM_LENGTH_2 * np.sin(theta1 + theta2) + r_y_data
    y = y.reshape(-1,1)
    pos = np.concatenate((x,y),axis=1)
    theta1 = theta1.reshape(-1,1)
    theta2 = theta2.reshape(-1,1)
    thetas = np.concatenate((theta1,theta2), axis=1)
    return train_test_split(pos, thetas, random_state=42)

def plot_normal_mix(pis, mus, sigmas, ax, label='', comp=True):
  """Plots the mixture of Normal models to axis=ax comp=True plots all
  components of mixture model
  """
  x = np.linspace(0.0, 2*np.pi, 250)
  final = np.zeros_like(x)
  for i, (weight_mix, mu_mix, sigma_mix) in enumerate(zip(pis, mus, sigmas)):
    temp = stats.norm.pdf(x, mu_mix, sigma_mix) * weight_mix
    final = final + temp
    if comp:
      ax.plot(x, temp, label='Normal ' + str(i))
  ax.plot(x, final, label='Mixture of Normals ' + label)
  ax.legend(fontsize=13)

N = 5000  # number of data points
D = 2  # number of features
K = 20  # number of mixture components

X_ph = tf.placeholder(tf.float32, [None, D])
y_ph = tf.placeholder(tf.float32, [None])

def neural_network(X):
  """loc, scale, logits = NN(x; theta)"""
  # 2 hidden layers with 15 hidden units
  hidden1 = slim.fully_connected(X, 15)
  hidden2 = slim.fully_connected(hidden1, 15)
  locs = slim.fully_connected(hidden2, K, activation_fn=None)
  scales = slim.fully_connected(hidden2, K, activation_fn=tf.exp)
  logits = slim.fully_connected(hidden2, K, activation_fn=None)
  return locs, scales, logits

# the model
locs, scales, logits = neural_network(X_ph)
cat = Categorical(logits=logits)
components = [Normal(loc=loc, scale=scale) for loc, scale
              in zip(tf.unstack(tf.transpose(locs)),
                     tf.unstack(tf.transpose(scales)))]
y = Mixture(cat=cat, components=components, value=tf.zeros_like(y_ph))

# data
X_train, X_test, y_train, y_test = gen_training(N)
y_train1 = y_train[:,0]
y_train2 = y_train[:,1]
y_test1 = y_test[:,0]
y_test2 = y_test[:,1]

# There are no latent variables to infer. Thus inference is concerned
# with only training model parameters, which are baked into how we
# specify the neural networks.
inference = ed.MAP(data={y: y_ph})
inference.initialize(var_list=tf.trainable_variables())

inference.initialize(var_list=tf.trainable_variables())

sess = ed.get_session()
tf.global_variables_initializer().run()

n_epoch = 2
train_loss = np.zeros(n_epoch)
test_loss = np.zeros(n_epoch)
for i in range(n_epoch):
  info_dict = inference.update(feed_dict={X_ph: X_train, y_ph: y_train1})
  train_loss[i] = info_dict['loss']
  test_loss[i] = sess.run(inference.loss,
                          feed_dict={X_ph: X_test, y_ph: y_test1})
  inference.print_progress(info_dict)

weights, pred_means, pred_std = \
    sess.run([tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: X_test})

tw0, tp0, ts0 = \
    sess.run([tf.nn.softmax(logits), locs, scales], feed_dict={X_ph: np.array([[3.,3.]])})

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 3.5))
plt.plot(np.arange(n_epoch), -test_loss / len(X_test), label='Test')
plt.plot(np.arange(n_epoch), -train_loss / len(X_train), label='Train')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Log-likelihood', fontsize=15)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
plot_normal_mix(tw0[0], tp0[0], ts0[0], axes, comp=False)
