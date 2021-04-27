import cppimport
import numpy as np
import seaborn as sns
import time

test = cppimport.imp("project")

if __name__ == '__main__':
    np.random.seed(1234)
    np.random.seed(1234)
    p = 2 
    theta = np.array([-3.0, 3.0]).reshape(p,-1)
    n = 200 
    x = np.array([np.random.normal(theta[0], 1, (n,1)),
                  np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)
    theta_0 = theta
    eta = 0.01/n * np.eye(p)
    alpha = 0.1 * np.eye(p)
    V = np.eye(p)*1
    niter = 500
    batch_size=50
    times_all = np.zeros(7)
    for i in range(7):
        t0 = time.time()
        samps_sghmc = test.sghmc("mixture_of_normals", eta, niter, alpha, theta_0, V, x, batch_size)
        t1 = time.time()
        times_all[i] = t1 - t0
    print(times_all.mean(),"s ±",times_all.std(), 
          "s per loop (mean ± std. dev. of 7 runs, 1 loop each)")