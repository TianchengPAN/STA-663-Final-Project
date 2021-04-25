import sys 
sys.path.append('/home/jovyan/work/STA-663-Final-Project/SGHMC')
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import pystan 
from sghmc.SGHMC_algorithm import sghmc,sghmc_cleaned

## example 2, where the data is in the distribution 0.5*N(mu_1,1)+0.5*N(mu_2,1),
##where mu_1=-3, and mu_2=3
def log_prior(theta):
    ## log density function for the prior distribution
    return(-(1/(2*10))*theta.T@theta)

def log_lik(theta, x):
    ## log likelihood for x|\theta
    return(np.log(0.5 * np.exp(-0.5*(theta[0]-x)**2) + 0.5* np.exp(-0.5*(theta[1]-x)**2)))
    
def U(theta, x, n, batch_size):
    return(-log_prior(theta) - (n/batch_size)*sum(log_lik(theta, x)))

        
# Automatic differentiation to get the gradient
gradU = jacobian(U, argnum=0)
    
# Set random seed
np.random.seed(1111)
# Set up the data
p = 2 #dimension of theta
theta = np.array([-3.0, 3.0]).reshape(p,-1)
n = 10000
x = np.array([np.random.normal(theta[0], 1, (n,1)),
                  np.random.normal(theta[1], 1, (n,1))]).reshape(-1,1)

    
## Initialize parameters and sample 
    
# Initialize mean parameters
#theta_0 = np.random.normal(size=(p,1))
theta_0 = theta # initialize at "true" value for testing
    
# Initialize tuning parameters:
# learning rate
eta = 0.01/n * np.eye(p)
# Friction rate
alpha = 0.1 * np.eye(p)
    
# Arbitrary guess at covariance of noise from mini-batching the data
V = np.eye(p)*1
niter = 100
batch_size=1000
    
# Run sampling algorithm
samps = sghmc(gradU, eta, niter, alpha, theta_0, V, x, batch_size)

sns.kdeplot(samps[0,:], samps[1,:]) # Plot the joint density
plt.title("Kernel density plot run by own algorithem")
plt.savefig('MixNorm_a.png')

