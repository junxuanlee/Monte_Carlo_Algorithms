import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import scipy.stats
from progressbar import ProgressBar

from numpy.random import default_rng
from numpy.random import RandomState

import sklearn
from celluloid import Camera
from IPython.display import HTML

from moviepy.editor import *

def generate_two_class_data(c, num_sample):
    rand = RandomState(123)
    
    mean0 = [-c,c]
    var0 = [[2,1],[1,2]]
    class0_x = rand.multivariate_normal(mean0, var0,num_sample)
    class0_y = np.zeros(class0_x.shape[0])

    mean1 = [c,-c]
    var1 = [[2,1],[1,2]]
    class1_x = rand.multivariate_normal(mean1, var1,num_sample)
    class1_y = np.ones(class1_x.shape[0])

    inputs = np.concatenate((class0_x, class1_x), axis=0)
    targets = np.concatenate((class0_y, class1_y), axis=0)

    inputs_shuffled, targets_shuffled = sklearn.utils.shuffle(inputs, targets)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,3))
    plt.tight_layout()

    ax.scatter(class0_x[:,0],class0_x[:,1])
    ax.scatter(class1_x[:,0],class1_x[:,1])
    ax.grid(True)
    ax.set_title("Class0 & Class1", color='m')
    
    return class0_x, class1_x, inputs_shuffled, targets_shuffled

def plot_final_decision_boundary_all(inputs_shuffled, z, class0_x, class1_x,titles):

    z_plot_all = []
    xx_all = []

    for i in range(len(z)):
        z_plot = []
        xx = []

        b = max(max(class0_x[:,0]),max(class0_x[:,1]),max(class1_x[:,0]),max(class1_x[:,1]))

        for k in range(len(inputs_shuffled)):
            z_plot.append((-(z[i][:,0][-1])*inputs_shuffled[k][0])/z[i][:,1][-1])
            xx.append(inputs_shuffled[k][0])

        z_plot_all.append(z_plot)
        xx_all.append(xx)

    plt.xlim(-(b+1), b+1)
    plt.ylim(-(b+1), b+1)

    for i in range(len(z)):
        plt.plot(xx_all[i], z_plot_all[i]);

    plt.legend(titles)
    plt.scatter(class0_x[:,0],class0_x[:,1], color='blue')
    plt.scatter(class1_x[:,0],class1_x[:,1], color='orange')
    plt.grid(True)
    plt.title("Class0 & Class1", color='m')

def plot_final_decision_boundary_single(inputs_shuffled, z, class0_x, class1_x):

    z_plot = []
    xx = []

    b = max(max(class0_x[:,0]),max(class0_x[:,1]),max(class1_x[:,0]),max(class1_x[:,1]))

    for k in range(len(inputs_shuffled)):
        z_plot.append((-(z[:,0][-1])*inputs_shuffled[k][0])/z[:,1][-1])
        xx.append(inputs_shuffled[k][0])

    plt.xlim(-(b+1), b+1)
    plt.ylim(-(b+1), b+1)

    plt.plot(xx, z_plot);
    plt.scatter(class0_x[:,0],class0_x[:,1], color='blue')
    plt.scatter(class1_x[:,0],class1_x[:,1], color='orange')
    plt.grid(True)
    plt.title("Class0 & Class1", color='m')

def animationlg(inputs_shuffled, z, class0_x, class1_x, c):

    fig = plt.figure()

    camera = Camera(fig)

    n = 0
    b = max(max(class0_x[:,0]),max(class0_x[:,1]),max(class1_x[:,0]),max(class1_x[:,1]))

    for w in range(z.shape[0]):
        z_plot = []
        xx = []
        for k in range(len(inputs_shuffled)):
            z_plot.append((-(z[:,0][w])*inputs_shuffled[k][0])/z[:,1][w])
            xx.append(inputs_shuffled[k][0])

        plt.xlim(-(b+1), b+1)
        plt.ylim(-(b+1), b+1)
        plt.plot(xx, z_plot, color='red');
        plt.scatter(class0_x[:,0],class0_x[:,1], color='blue')
        plt.scatter(class1_x[:,0],class1_x[:,1], color='orange')
        plt.grid(True)
        plt.title("Class0 & Class1", color='m')

        camera.snap()

    plt.close()

    return camera

class SIR_logistic_regression:
    def __init__(self,inputs_shuffled,targets_shuffled, T, L, Q, R, z_init):
        self.inputs_shuffled = inputs_shuffled
        self.targets_shuffled = targets_shuffled
        self.T =T
        self.L = L
        self.Q = Q
        self.R = R
        self.xx = self.targets_shuffled.copy()
        
        self.zs = np.zeros((self.T,self.L,2))
        self.ws = np.zeros((self.T,self.L))
        self.z = np.zeros((self.T,2))
       
        self.z_init = z_init
        self.zs[0] = self.z_init
        
        self.ws[:1] = 1/self.L
        
        self.rand = RandomState(123)
        
        self.ess = np.zeros(self.T)
        
        self.targets = self.targets_shuffled.copy()

        self.e = np.zeros(self.targets.shape) 
        
        self.mean = None
        self.H_hat = None
        self.mu = None
        self.w_sum = None
    
    def calculate_previous_mean(self,t):
        self.mean = 0
        for k in range(len(self.ws[t-1])):
            self.mean = self.mean + self.ws[t-1][k]*self.zs[t-1][k]
        self.mean = np.expand_dims(self.mean,axis=-1)        
    
    def draw_samples(self,t,l):
        self.zs[t,l] = self.rand.multivariate_normal(self.zs[t-1,l], self.Q)
        
    def update_weight(self,t,l):
        self.mu = 1/(1+ np.exp(-self.zs[t,l]@self.inputs_shuffled[t]))
        p = scipy.stats.norm.pdf(self.xx[t], self.mu, self.R)
        
        #Update weights
        self.ws[t,l] = self.ws[t-1,l]*p   
        
    def normalize_weights(self,t):
        for l in range(self.L):
            self.ws[t,l] = self.ws[t,l]/self.w_sum  
            
    def calculate_ESS(self,t):
        ess_sum = 0
        for l in range(self.L):
            ess_sum += (self.ws[t,l])**2

        self.ess[t] = 1/ess_sum      
        
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
        
    def train(self, resample=False, num=1000):
        pbar = ProgressBar()
        for t in pbar(range(1,self.T)):

            self.w_sum = 0

            #self.calculate_previous_mean(t)

            for l in range(self.L):

                self.draw_samples(t,l)

                #self.calculate_H_hat(t,l)        

                self.update_weight(t,l)

                self.w_sum += self.ws[t,l]

            self.normalize_weights(t)

            self.calculate_ESS(t)

            if self.ess[t] < num:
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
            
    def calculate_loss(self):
        for i in range(len(self.targets)):
            xx = np.expand_dims(self.inputs_shuffled[i],axis=-1)

            zz = np.expand_dims(self.z[i],axis=-1)
            hu = 1/(1+ np.exp(-zz.T@xx))

            self.e[i] = (self.targets[i] - hu)**2       
            
    def plot(self):
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12,8))
        plt.tight_layout()

        ax[0,0].plot(self.z[:,0])
        ax[0,0].grid(True)
        ax[0,0].set_title("Coefficient a0", color='m')
        #ax[0,0].plot(range(N), A[:,0], color='m')

        ax[0,1].plot(self.z[:,1])
        ax[0,1].grid(True)
        ax[0,1].set_title("Coefficient a1", color='m')
        #ax[0,1].plot(range(N), A[:,1], color='m')

        ax[1,0].plot(self.e)
        ax[1,0].grid(True)
        ax[1,0].set_title("loss e0", color='m')
        #ax[1,0].set_ylim([0, 0.8])

        ax[1,1].plot(self.ess)
        ax[1,1].grid(True)
        ax[1,1].set_title("ess", color='m')
        
