import numpy as np
import matplotlib.pyplot as plt
np.random.seed(8)
X = np.random.randn(1000,1)
y = 2*(X**3) + 4.6*np.random.randn(1000,1)
def wm(point, X, var):
    m = X.shape[0]
    w = np.mat(np.eye(m)) # Initialising W as an identity matrix.
    for i in range(m):
        xi = X[i]
        d = (-2 * var * var)
        w[i, i] = np.exp(np.dot((xi-point), (xi-point).T)/d)
    return w

def predict(X, y, point, var):
    # m = number of training examples.
    # Appending a cloumn of ones in X to add the bias term.
    m = X.shape[0]
    X_ = np.append(X, np.ones(m).reshape(m,1), axis=1)
    point_ = np.array([point, 1])
    w = wm(point_, X_, var) # Calculating the weight matrix
    beta = np.linalg.pinv(X_.T*(w * X_))*(X_.T*(w * y))
    pred = np.dot(point_, beta)
    return pred

def plot_predictions(X, y, var, nval):
    # nval --> number of points to be predicted. var --> the variance
    X_test = np.linspace(-3, 3, nval)
    pred = []
    for point in X_test:
        p = predict(X, y, point, var)
        pred.append(p)

    X_test = np.array(X_test).reshape(nval,1)
    pred = np.array(pred).reshape(nval,1)

    plt.plot(X, y, 'b.') #b. dots, b- solid line
    plt.plot(X_test, pred, 'r.')
    plt.show()

plot_predictions(X, y, 0.5, 100)