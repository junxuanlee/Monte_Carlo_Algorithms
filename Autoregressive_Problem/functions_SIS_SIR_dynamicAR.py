import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats
from progressbar import ProgressBar

from numpy.random import default_rng
from numpy.random import RandomState

def plot(predictions, titles, T, A, ess,colour):
    
    e = np.zeros((len(predictions),T,2))
    
    for k in range(len(predictions)):
        for i in range(len(predictions[k])):
            e[k,i,0] = (A[:,0][i] - predictions[k][:,0][i])**2
            e[k,i,1] = (A[:,1][i] - predictions[k][:,1][i])**2

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14,10))
    plt.tight_layout()

    for q in range(len(predictions)):
        ax[0,0].plot(predictions[q][:,0], color=colour[q])
    ax[0,0].plot(range(T), A[:,0], color='m')
    ax[0,0].legend(titles)
    ax[0,0].grid(True)
    ax[0,0].set_title("Coefficient a0", color='m')



    for q in range(len(predictions)):
        ax[0,1].plot(predictions[q][:,1], color=colour[q])
    ax[0,1].plot(range(T), A[:,1], color='m')
    ax[0,1].legend(titles)
    ax[0,1].grid(True)
    ax[0,1].set_title("Coefficient a1", color='m')


    for q in range(len(predictions)):
        ax[1,0].plot(e[q][:,0], color=colour[q])
    ax[1,0].grid(True)
    ax[1,0].legend(titles)
    ax[1,0].set_title("loss e0", color='m')

    for q in range(len(predictions)):
        ax[1,1].plot(e[q][:,1], color=colour[q])
    ax[1,1].grid(True)
    ax[1,1].legend(titles)
    ax[1,1].set_title("loss e1", color='m')

    for s in range(len(ess)):
        ax[2,0].plot(ess[s], color=colour[s])
    ax[2,0].legend([titles[0],titles[1]])
    ax[2,0].grid(True)
    ax[2,0].set_title("ess", color='m')

    for s in range(len(ess)):
        ax[2,1].plot(ess[s], color=colour[s])
    ax[2,1].legend([titles[0],titles[1]])
    ax[2,1].grid(True)
    ax[2,1].set_title("ess", color='m')


class dynamic_2nd_AR_:
    def __init__(self,N):
        self.N = N
        self.ex = None
        self.A = None
        self.S = None
        
    def generate_signal(self):
        # Gaussian random numbers as an excitation signal
        self.ex = np.random.randn(self.N)

        # Second order AR Process with coefficients slowly changing in time
        a0 = np.array([1.2, -0.4])
        self.A = np.zeros((self.N,2))
        alpha = 0.1

        for n in range(self.N):
            self.A[n,0] = a0[0] + alpha * np.cos(2*np.pi*n/self.N)
            self.A[n,1] = a0[1] + alpha * np.sin(np.pi*n/self.N)

        self.S = self.ex.copy();
        for n in range(2, self.N):
            x = np.array([self.S[n-1], self.S[n-2]])
            self.S[n] = np.dot(x, self.A[n,:]) + self.ex[n]
            
    def plot(self):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,4))
        plt.tight_layout()

        ax[1,0].plot(range(self.N), self.A[:,0])
        ax[1,0].grid(True)
        ax[1,0].set_title("s0 state", color='m')

        ax[1,1].plot(range(self.N), self.A[:,1], color='m')
        ax[1,1].grid(True)
        ax[1,1].set_title("s1 state", color='m')

        ax[0,0].plot(range(self.N), self.ex)
        ax[0,0].grid(True)
        ax[0,0].set_title("Random Excitation Signal")

        ax[0,1].plot(range(self.N), self.S, color='m')
        ax[0,1].grid(True)
        ax[0,1].set_title("Dynamic 2nd Order AR signal")
       
class particle_filter:
    def __init__(self,T,L,Q,R,S,z_init, A):
        self.T = T
        self.L = L
        self.Q = Q
        self.R = R
        self.z_init = z_init
        self.S = S.copy()
        self.A = A
        
        self.zs = np.zeros((self.T, self.L,2))
        self.ws = np.zeros((self.T, self.L))
        self.z = np.zeros((self.T,2))
        
        #Intial two states (guess)
        self.zs[0] = self.z_init
        self.zs[1] = self.z_init   
        
        #Intial two weights (even weights)
        self.ws[:2] = 1/self.L
        
        self.ess = np.zeros(self.T)
        self.e = np.zeros((self.T,2))

        self.rand = RandomState(123)
        
    def draw_sample(self,t,l):
        self.zs[t,l] = self.rand.multivariate_normal(self.zs[t-1,l], self.Q)
        
    def update_weight(self,t,l):
        mu = self.zs[t,l][0]*self.S[t-1] + self.zs[t,l][1]*self.S[t-2]
        p = scipy.stats.norm.pdf(self.S[t], mu, self.R)
        
        self.ws[t,l] = self.ws[t-1,l]*p   
        
    def calculate_ESS(self,t):
        ess_sum = 0
        for l in range(self.L):
            ess_sum += (self.ws[t,l])**2

        self.ess[t] = 1/ess_sum        

    def normalize_weights(self,t,w_sum):
        for l in range(self.L):
            self.ws[t,l] = self.ws[t,l]/w_sum    
        
    def resampling(self,t):
        i = 0

        c = np.zeros(self.L)
        c[0] = 0

        u = np.zeros(self.L)
        u[0] = self.rand.uniform(low=1.e-300, high=1/self.L)

        for l in range(1,self.L):
            c[l] = c[l-1] + self.ws[t,l]

        for l in range(self.L-1):
            u[l+1] = u[l] + 1/self.L

        s = []
        for l in range(self.L):
            while u[l] > c[i]:
                if i == self.L-1:
                    break
                i += 1

            s.append(self.zs[t,i])

        for l in range(self.L):
            self.zs[t,l] = s[l]

        self.ws[t] = 1/self.L     

    def train(self, resample = False):
        pbar = ProgressBar()
        for t in pbar(range(2,self.T)):
            
            w_sum = 0
            for l in range(self.L):

                self.draw_sample(t, l)
                self.update_weight(t, l)

                w_sum += self.ws[t,l]

            self.normalize_weights(t, w_sum)

            self.calculate_ESS(t)
            
            if resample:
                self.resampling(t)
            
    def predict(self):

        for t in range(self.T):
            a0 = 0
            a1 = 0
            
            for l in range(self.L):
                a0 = a0 + self.ws[t,l]*self.zs[t,l,0]
                a1 = a1 + self.ws[t,l]*self.zs[t,l,1]

            self.z[t,0] = a0
            self.z[t,1] = a1   
            
    def plot(self):
        for i in range(len(self.z)):
            self.e[i,0] = (self.A[:,0][i] - self.z[:,0][i])**2
            self.e[i,1] = (self.A[:,1][i] - self.z[:,1][i])**2

        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(13,10))
        plt.tight_layout()

        ax[0,0].plot(self.z[:,0])
        ax[0,0].grid(True)
        ax[0,0].set_title("Coefficient a0", color='m')
        ax[0,0].plot(range(self.T), self.A[:,0], color='m')

        ax[0,1].plot(self.z[:,1])
        ax[0,1].grid(True)
        ax[0,1].set_title("Coefficient a1", color='m')
        ax[0,1].plot(range(self.T), self.A[:,1], color='m')

        ax[1,0].plot(self.e[:,0])
        ax[1,0].grid(True)
        ax[1,0].set_title("loss e0", color='m')
        #ax[1,0].set_ylim([0, 0.8])

        ax[1,1].plot(self.e[:,1])
        ax[1,1].grid(True)
        ax[1,1].set_title("loss e1", color='m')
        #ax[1,1].set_ylim([0, 1.2])

        ax[2,0].plot(self.ess)
        ax[2,0].grid(True)
        ax[2,0].set_title("ess", color='m')

        ax[2,1].plot(self.ess)
        ax[2,1].grid(True)
        ax[2,1].set_title("ess", color='m')
