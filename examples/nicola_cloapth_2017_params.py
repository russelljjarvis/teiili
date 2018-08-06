# ## IZFORCE CLASSIFIER WITH ADAPTATION 
# clear all
# close all
# clc 
import numpy as np
from  numpy import round
from  numpy import zeros
from numpy.random import randn
from numpy.random import rand
from numpy import sqrt
from numpy import mod
from numpy import pi
from numpy import cos
from numpy import sin
from numpy import ceil
from numpy import eye
import ipdb

# Initialize RNG seed
np.random.seed(123456789)


T = 1000 #Total time in seconds
dt = 0.01 #integration time step 
nt = int(round(T/dt))
N =  20  #number of neurons 
## Izhikevich Parameters
C = 250 # pF
vr = -60 
b = 0 # pS
k = 2.5 
vpeak = 30 
vreset = -65
vt = vr+40-(b/k) #threshold 
Er = 0 #Reversal Potential 
u = zeros((N,1)) 
a = 0.01 # 1/tauadapt = 1/ms -> tauiadapt = 100 ms
d = 200 
tr = 2 
td = 20 
p = 0.1 #sparsity 
G =6*10**3 #scale weight matrix  
BIAS = 1000 #Bias current
OMEGA =  G*(randn(N,N))*(rand(N,N)<p)/(p*sqrt(N))
## Initialize currents, FORCE method, other parameters
IPSC = zeros((N,1)) #post synaptic current 
h = zeros((N,1))
r = zeros((N,1))
hr = zeros((N,1))
JD = zeros((N,1))
BPhi = zeros((N,1)).reshape((20))

#-----Initialization---------------------------------------------
v_init = vr+(vpeak-vr)*rand(N,1) #initial distribution
v = v_init
# v_ = v #These are just used for Euler integration, previous time step storage
IPSC = zeros((N,1))
Q = 5*10**3 #Scale feedback term, Q in paper
E = (2*rand(N,1)-1)*Q #scale feedback term
E = E.reshape((N))
WE2 = 5*10**2 #scale input weights
Psi = 2*pi*rand(N,1) 
Ein = np.column_stack((cos(Psi),sin(Psi)))*WE2

from scipy.io import loadmat
lineardata = loadmat('lineardata.mat')
## lineardata.mat corresponds to linear boundaries while nonlineardata.mat corresponds to nonlinear boundaries 

#Load the data set for classification.  P contains class, z contains the points.  
inputfreq = 4 #Present a data point 
j =0
Xin = zeros((nt,2)) 
zx = zeros((nt,1)) 
nx = round(1000/(dt*inputfreq)) 
# zx = 0.5*abs(sin(2*pi*(1:1:nt)*dt*inputfreq/2000)) 
zx = 0.5*abs(sin(2*pi*np.arange(1,nt+1,1)*dt*inputfreq/2000)) 

## Create supervisor and inputs.  
j = 1 
k2 = 0
# for i =1:1:nt 
for i in range(0,nt,1): 
    # zx(i) = zx(i)*(2*P(j)-1)*mod(k2,2)
    # Xin(i,:) = [z(j,1),z(j,2)]*mod(k2,2)
    zx[i] = zx[i]*(2*lineardata['P'][j]-1.0)*mod(k2,2)
    Xin[i,:] = np.array([lineardata['z'][j,0],lineardata['z'][j,1]]) * mod(k2,2)
    if mod(i+1,nx)==1 :
        k2 = k2 + 1 
        j = int(ceil(rand()*2000))
    # end
# end
## initialization 
z = 0 
tspike = zeros((nt,2))
ns = 0
## RLS parameters.  
Pinv = eye(N)*30
step = 20
imin = round(10/dt)
icrit = round(400000/dt)
current = zeros((nt,1)) 
current_r = zeros((nt,N)) 
current_v = zeros((nt,N)) 
RECB = zeros((nt,5))
REC = zeros((nt,10))
i=1
# ## SIMULATION
# tic
# ilast = i 
# for i = ilast:1:nt 
# ## EULER INTEGRATE
# I = IPSC + E*z + Ein*(Xin(i,:)') + BIAS 
# v = v + dt*(( k.*(v-vr).*(v-vt) - u + I))/C  # v(t) = v(t-1)+dt*v'(t-1)
# u = u + dt*(a*(b*(v_-vr)-u)) #same with u, the v_ term makes it so that the integration of u uses v(t-1), instead of the updated v(t)

# ## 
# index = find(v>=vpeak)
# if length(index)>0
# JD = sum(OMEGA(:,index),2) #compute the increase in current due to spiking  
# #tspike(ns+1:ns+length(index),:) = [index,0*index+dt*i]
# ns = ns + length(index) 
# # end


# if tr == 0 
#     IPSC = IPSC*exp(-dt/td)+   JD*(length(index)>0)/(td)
#     r = r *exp(-dt/td) + (v>=vpeak)/td
# else
# IPSC = IPSC*exp(-dt/td) + h*dt
# h = h*exp(-dt/tr) + JD*(length(index)>0)/(tr*td)  #Integrate the current

# r = r*exp(-dt/td) + hr*dt 
# hr = hr*exp(-dt/tr) + (v>=vpeak)/(tr*td)
# # end


# ## Apply RLS 
#  z = BPhi'*r
#  err = z - zx(i)
# if mod(i,step)==1
# if i > imin 
#  if i < icrit 
#    cd = Pinv*r
#    BPhi = BPhi - (cd*err')
#    Pinv = Pinv -((cd)*(cd'))/( 1 + (r')*(cd))
#  # end 
# # end 
# # end




# ## COMPUTE S, APPLY RESETS
# u = u + d*(v>=vpeak)  #implements set u to u+d if v>vpeak, component by component. 
# v = v+(vreset-v).*(v>=vpeak) #implements v = c if v>vpeak add 0 if false, add c-v if true, v+c-v = c
# v_ = v  # sets v(t-1) = v for the next itteration of loop
# REC(i,:) = [v(1:5)',u(1:5)'] 
# current(i,:) = z 
# current_r(i,:) = r 
# current_v(i,:) = v 
# RECB(i,:)=BPhi(1:5)
# if mod(i,round(100/dt))==1 
# drawnow

# figure(2)
# xlim_min = max([i-1e5 1])
# plot(dt*(xlim_min:1:i),current(xlim_min:1:i,1)), hold on 
# plot(dt*(xlim_min:1:i),zx(xlim_min:1:i),'k--'), hold off
# ylim([-0.6,0.6])
# xlim([dt*i-1000,dt*i])
# xlabel('Time')
# ylabel('Network Response')
# leg# end('Network Output','Target Signal')
# ## plot supervisor and inputs
# figure(4)
# if i>xlim_min
# limits = xlim_min:i#(1:1e5)+i*1e5
# subplot(3,1,2)
# plot(limits,zx(limits(1):limits(# end)))
# xlim([limits(1) limits(# end)])
# subplot(3,1,3)
# plot(limits,Xin(limits(1):limits(# end),1))
# xlim([limits(1) limits(# end)])
# subplot(3,1,1)
# plot(limits,Xin(limits(1):limits(# end),2))
# xlim([limits(1) limits(# end)])
# #pause(0.5)
# # end
# # end   
# # end
# ## 

plot_it = False
if plot_it:
    from matplotlib import pyplot as plt
    plt.plot(Xin[:,0])
    plt.plot(Xin[:,1])
    plt.plot(zx)
    
    plt.show()
