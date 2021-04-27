import cppimport
import numpy as np
import seaborn as sns
import time

code = cppimport.imp("project")

if __name__ == '__main__':
    np.random.seed(1234)
    # Don't actually need 'data' in this example, just use
    # it as a place-holder to fit into our function.
    n = 100
    x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
    # Set up start values and tuning params
    theta_0 = np.array([0.0]) # Initialize theta
    p = theta_0.shape[0]
    eta = 0.001 * np.eye(p) # make this small
    alpha = 0.01 * np.eye(p)
    V = np.eye(p)
    batch_size = n # since we're not actually using the data, don't need to batch it
    niter = 50000000 # TONS of iterations, to get nice figure
    # run SGHMC sampler
    samps_sghmc = code.sghmc("fig1", eta, niter, alpha, theta_0, V, x, batch_size)
    # save to a file
    np.save("samps_sghmc.npy",samps_sghmc)
    # ADD THIS to save plot to view
    # plot the samples from the algorithm and save to a file
    kdeplt = sns.kdeplot(samps_sghmc.reshape(-1))
    fig = kdeplt.get_figure()
    fig.savefig('Example1_a.png')