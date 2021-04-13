import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def computeCost(X, y, b=0, w=np.zeros((1,1))):     
      
    m = y.size
    
    y_hat = np.dot(X, w) + b
    L = np.square(y_hat - y)
    J = 1 / (2 * m) * np.sum(L)
    
    return J


def fancy_plots(X, y, bs, ws, new_b, new_w):
    # Create grid coordinates for plotting
    B0 = np.linspace(-10, 10, 50)
    B1 = np.linspace(-1, 4, 50)
    xx, yy = np.meshgrid(B0, B1)
    Z = np.zeros((B0.size, B1.size))

    # Calculate Z-values (Cost) based on grid of coefficients
    for (i, j), v in np.ndenumerate(Z):
        Z[i, j] = computeCost(X, y, b=xx[i, j], w=yy[i, j])

    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    # Contour plot
    CS = ax1.contour(xx, yy, Z, np.logspace(-2, 3, 20), cmap=plt.cm.jet)
    for i in range(len(ws)):
        if i == 0:
            ax1.scatter(bs[i], ws[i], s=5, c='r')
        if (i+1)%150==0:
            ax1.scatter(bs[i], ws[i], s=5, c='r')
    ax1.scatter(new_b, new_w, c='r')

    # 3d plot
    ax2.plot_surface(xx, yy, Z, rstride=1, cstride=1, alpha=0.6, cmap=plt.cm.jet)
    ax2.set_zlabel(r'$J$', fontsize=17)
    ax2.set_zlim(Z.min(), Z.max())
    ax2.view_init(elev=15, azim=230)

    # labeling parameters which is the same for both plots
    for ax in fig.axes:
        ax.set_xlabel(r'$b$', fontsize=17)
        ax.set_ylabel(r'$w_1$', fontsize=17)