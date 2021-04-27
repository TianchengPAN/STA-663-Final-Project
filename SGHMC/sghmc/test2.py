import cppimport
import numpy as np
import seaborn as sns
import time

test = cppimport.imp("project")

if __name__ == '__main__':
    np.random.seed(1234)
    n = 100
    x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
    theta_0 = np.array([0.0]) 
    p = theta_0.shape[0]
    eta = 0.001 * np.eye(p) 
    alpha = 0.01 * np.eye(p)
    V = np.eye(p)
    batch_size = n
    niter = 50000 
    
    times_all = np.zeros(7)
    for i in range(7):
        t0 = time.time()
        samps_sghmc = test.sghmc("fig1", eta, niter, alpha, theta_0, V, x, batch_size)
        t1 = time.time()
        times_all[i] = t1 - t0
    print(times_all.mean(),"s ±",times_all.std(), 
          "s per loop (mean ± std. dev. of 7 runs, 1 loop each)")