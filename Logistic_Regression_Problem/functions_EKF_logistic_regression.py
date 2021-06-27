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

def plot_final_decision_boundary(inputs_shuffled, theta0_list, theta1_list, class0_x, class1_x):

    z = []
    xx = []

    b = max(max(inputs_shuffled[:,0]),max(inputs_shuffled[:,1]))

    for k in range(len(inputs_shuffled)):
        z.append((-(theta0_list[-1])*inputs_shuffled[k][0])/theta1_list[-1])
        xx.append(inputs_shuffled[k][0])


    plt.xlim(-(b+1), b+1)
    plt.ylim(-(b+1), b+1)
    plt.plot(xx, z, color='red');
    plt.scatter(class0_x[:,0],class0_x[:,1], color='blue')
    plt.scatter(class1_x[:,0],class1_x[:,1], color='orange')
    plt.grid(True)
    plt.title("Class0 & Class1", color='m')

def animation(inputs_shuffled, theta0_list, theta1_list, class0_x, class1_x, c):

    fig = plt.figure()

    camera = Camera(fig)

    n = 0
    
    b = max(max(class0_x[:,0]),max(class0_x[:,1]),max(class1_x[:,0]),max(class1_x[:,1]))

    for w in range(len(theta0_list)):
        z = []
        xx = []
        for k in range(len(inputs_shuffled)):
            z.append((-(theta0_list[w])*inputs_shuffled[k][0])/theta1_list[w])
            xx.append(inputs_shuffled[k][0])


        plt.xlim(-(b+1), b+1)
        plt.ylim(-(b+1), b+1)
        plt.plot(xx, z, color='red');
        plt.scatter(class0_x[:,0],class0_x[:,1], color='blue')
        plt.scatter(class1_x[:,0],class1_x[:,1], color='orange')
        plt.grid(True)
        plt.title("Class0 & Class1", color='m')

        camera.snap()

    plt.close()

    return camera



class extended_kalman_filter:
    def __init__(self,inputs_shuffled,targets_shuffled,theta_n1_n1,P_n1_n1,Q,R):
        self.inputs_shuffled = inputs_shuffled
        self.targets_shuffled = targets_shuffled
        self.theta_n1_n1 = theta_n1_n1
        self.P_n1_n1 = P_n1_n1
        self.Q = Q
        self.R = R
        
        self.N = self.inputs_shuffled.shape[0]
        self.y = self.targets_shuffled.copy()
        self.theta0_list = []
        self.theta1_list = []
        self.error = []
        
        self.theta_n_n1 = None
        self.P_n_n1 = None
        self.H_hat = None
        self.k_n = None
        self.theta_n_n = None
        self.P_n_n = None
        

    def calculate_h_hat(self,xs):   
        x0 = xs[0]
        x1 = xs[1]

        theta0 = self.theta_n_n1[0][0]
        theta1 = self.theta_n_n1[1][0]

        num0 = x0*np.exp(-x0*theta0 - x1*theta1)
        denum0 = (np.exp(-x0*theta0 - x1*theta1) + 1)**2
        dyd0 = num0/denum0

        num1 = x1*np.exp(-x1*theta1 - x0*theta0)
        denum1 = (np.exp(-x1*theta1 - x0*theta0) +1)**2
        dyd1 = num1/denum1

        self.H_hat = np.array([[dyd0, dyd1]])
        
    def predict(self):
        self.theta_n_n1 = self.theta_n1_n1.copy()
        self.P_n_n1 = self.P_n1_n1 + self.Q

    def update(self):
        num = self.P_n_n1@self.H_hat.T
        denum = self.R + self.H_hat@self.P_n_n1@self.H_hat.T

        self.k_n = num/denum

        self.theta_n_n = self.theta_n_n1 + self.k_n*self.e_n
        self.P_n_n = (np.identity(2) - self.k_n@self.H_hat)@self.P_n_n1

        self.theta_n1_n1 = self.theta_n_n.copy()
        self.P_n1_n1 = self.P_n_n.copy()

    def calculate_loss(self,inputs,y):
        xx = np.expand_dims(inputs,axis=-1)
        hu = 1/(1+ np.exp(-self.theta_n_n1.T@xx))

        self.e_n = (y - hu)[0][0]   
        self.error.append(self.e_n**2)
        
    def plot(self):
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,3))
        ax[0].plot(self.theta0_list)
        ax[1].plot(self.theta1_list)
        ax[2].plot(self.error)
        
    def train(self):
        for i in range(self.N):
            #Predict
            self.predict()

            self.calculate_h_hat(self.inputs_shuffled[i])

            self.calculate_loss(self.inputs_shuffled[i],self.y[i])

            #Update  
            self.update()
            
            self.theta0_list.append(self.theta_n_n[0][0])
            self.theta1_list.append(self.theta_n_n[1][0])

    
