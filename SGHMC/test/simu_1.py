import sys 
sys.path.append('/home/jovyan/work/STA-663-Final-Project/SGHMC')
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import autograd.numpy as np
from autograd import jacobian
import pystan 
from sghmc.SGHMC_algorithm import sghmc,sghmc_cleaned

## the first simulation exapmle is as described as in the origianl paper 
## , where the potential energy is $U(\theta)=-2\theta^2+\theta^4$


def U(theta):
    """
    the simulation function
    """
    return(-2*theta**2 + theta**4)
# True gradient
gradU = jacobian(U, argnum=0)
# stochastic gradient
def noisy_gradU(theta, x, n, batch_size):
    ''' stochastic gradient \Delta\tilde{U}(\theta)=\Delta U(\theta)+N(0,4)
    Extra args (x, n, batch_size) for compatibility with sghmc()'''
    return -4*theta + 4*theta**3 + np.random.normal(0,2)

# Set random seed
np.random.seed(1111)
n = 1000
x = np.array([np.random.normal(0, 1, (n,1))]).reshape(-1,1)
# Set up start values and tuning params
theta_0 = np.array([0.0]) # Initialize theta
p = theta_0.shape[0]
eps=0.1
eta = eps**2 * np.eye(p) # make this small
alpha = eps * np.eye(p)
V = np.eye(p)*0
batch_size = 1
niter = 2000
# run SGHMC sampler
samps_sghmc = sghmc_cleaned(noisy_gradU, eta, niter, alpha, theta_0, V, x, batch_size)
    
# plot the samples from the algorithm and save to a file
sns.kdeplot(samps_sghmc.reshape(-1)) # Plot the joint density
plt.title('Kernel density plot run by own algorithem', fontsize=14)
plt.savefig('Example1_a.png')

