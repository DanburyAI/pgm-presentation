import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression

def fake_data():
    x_train = np.linspace(-3, 3, num=50)
    y_train = x_train ** 2 + np.random.normal(0, 0.4, size=50)
    x_train = x_train.astype(np.float32).reshape((50, 1))
    y_train = y_train.astype(np.float32).reshape((50, 1))
    return x_train, y_train

def make_feats(x_train):
    x_squared_train = x_train ** 2
    return np.concatenate((x_train, x_squared_train), axis=1)

def plot_result(clf, x_train, y_train):
    x_test = np.linspace(-3, 3, num=50)
    y_test = clf.predict(make_feats(x_test.reshape(-1,1)))
    plt.plot(y_test)

def plot_energy_surface(clf):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_test = np.linspace(-1.0, 1.0, num=100)
    y_test = np.linspace(-1.0, 1.0, num=100)
    x_test, y_test = np.meshgrid(x_test, y_test)
    b = clf.intercept_[0]
    w1 = clf.coef_[0][0]
    w2 = clf.coef_[0][1]
    z = (0.5)*(((b + w1*x_test + w2*(x_test**2)) - y_test)**2) 
    surf = ax.plot_surface(x_test, y_test, z, cmap=cm.coolwarm,
                                              linewidth=0, antialiased=False)
    plt.show()

x_train, y_train = fake_data()
x_train = make_feats(x_train.reshape(-1,1))
y_train = y_train.reshape(-1,1)

clf = LinearRegression(fit_intercept=True)
clf.fit(x_train, y_train)


