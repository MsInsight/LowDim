#! /usr/bin/env python
# By Sara Salha  
# Nov 2013

'''
This is a helper code.
The main code studies the non-convexity of the modulus constraint, show the iterates in DM, AP and APR'''


import numpy as np                    
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from sys import argv, exit
from copy import deepcopy

# create the energy landscape:
def energy_landscape(a, b, d0, normr, normf, fstate, known):
  #x = np.linspace(0, a, b)
  #y = x#np.linspace(-a/2, a/2, b)
  x = np.linspace(-a/2, a/2, b)
  y = np.linspace(-a/2, a/2, b)
  epsilon_m = np.zeros([np.size(x), np.size(y)])#, np.size(z)])
  rho_t = np.zeros(d0)
  for i in range(len(x)):
    rho_t[0] = x[i]
    for j in range(len(x)):
      rho_t[1] = y[j]
      rho_t[2] = normr-x[i]-y[j]
      mod1 = np.abs( np.fft.fft( rho_t ))
      epsilon_m[i,j] = np.sum(np.power(mod1[known]-fstate[known],2))/normf
  return epsilon_m

# Project energy
def projectE(rho, fstate, known, normf, normr):
  rhop = np.fft.fft(rho)
  err = np.sum(np.power(np.abs(rhop[known])-fstate[known], 2))/normf
  phase = np.angle(rhop)
  rhop2 = fstate*np.exp(1j*phase)
  rho2 = np.real(np.fft.ifft(rhop2))
  normr2 = 1.0*normr/np.sum(rho2)
  rho2 *=normr2
  return rho2, err

# Project Geometry
def projectG(rho, support, normr):
    rho[support] = 0
    #rho[np.where(rho<0)] = 0
    normr1 = 1.0*normr/np.sum(rho)
    rho *=normr1
    return rho

def computeshift(kk, rho0, kk0):
       a1 = 1.0*(kk[0] - kk0)/(rho0[0])
       a2 = 1.0*(kk[1] - kk0)/(rho0[1])
       b1 = 1.0*(kk[0]-a1*rho0[0])
       b2 = 1.0*(kk[1]-a2*rho0[1])
       return a1, b1, a2, b2

def convertcoord(x1,y1, a1, b1, a2, b2):
    xc1, yc1 = int(a1*x1+b1), int(a2*y1+b2)
    return xc1,yc1

def plotellipsoids(a, b, normr, fstate):
   x = np.linspace(-a/2, a/2, b)
   y = np.linspace(-a/2, a/2, b)
   # Find the convex set
   yy,xx = np.meshgrid(x,y)
   epsilon = 1.00
   rho = np.zeros([np.size(x), np.size(y)])
   zz = normr-xx-yy
   mm = np.size(fstate)
   mask = 0*xx
   for k in range(mm):
       tt = (xx*xx+yy*yy+zz*zz+2*xx*yy*np.cos(2*np.pi*k*(0-1)/mm)+2*xx*zz*np.cos(2*np.pi*k*(0-2)/mm)+2*zz*yy*np.cos(2*np.pi*k*(1-2)/mm))
       e1 = fstate[k]**2
       tt[np.where(tt<=e1-epsilon)] = 0
       tt[np.where(tt>=e1+epsilon)] = 0
       tt[np.where(tt!=0)] = k
       mask += tt
   return mask

# Alternate projection
def AP(ite, rho0, support, fstate, known, normf, normr, a1, b1, a2, b2):
  trace = np.zeros([ite,3]) # track the iterate, and error
  rho1 = deepcopy(rho0)
  for i in range(ite):
     #d1 = projectG(rho1, support, normr)
     d1, err= projectE(rho1, fstate, known, normf, normr)
     d2 = projectG(d1, support, normr)
     trace[i, 0:2]  = d2[0:2]
     trace[i, 2] = err
     rho1 = deepcopy(d2)
  x1, y1 = convertcoord(rho1[0] ,rho1[1], a1, b1, a2, b2)
  print ('Last AP coord (%d, %d) with err %0.5f' %(x1, y1, err)) 
  return trace

def DM(ite, rho0, support, fstate, known, normf, normr, a1, b1, a2, b2):
  trace = np.zeros([ite,3]) # track the iterate, and error
  rho1 = deepcopy(rho0)
  d1, d2, epsilon = 1000,  0, 0.0001
  count = 0
  while (np.sum(np.abs(d1-d2))>epsilon and count <ite-1):
     d1   = projectG(rho1, support, normr)
     d2, err  = projectE(2*d1-rho1, fstate, known, normf, normr) 
     rho2 = rho1+d2-d1
     trace[count, 0:2]  = rho2[0:2]
     trace[count, 2] = err
     rho1 = deepcopy(rho2)
     count+=1
  # AP
  rho2 = projectG(rho2, support, normr)
  rho2, err= projectE(rho2, fstate, known, normf, normr)
  rho2 = projectG(rho2, support, normr)
  trace[count, 0:2]  = rho2[0:2]
  trace[count, 2] = err
  x1, y1 = convertcoord(rho2[0] ,rho2[1], a1, b1, a2, b2)
  print ('Last DM coord (%d, %d) with err %0.5f' %(x1, y1, err)) 
  return trace
  
def hio(ite, rho0, support, fstate, known, normf, normr):
  trace = np.zeros([ite,3]) # track the iterate, and error
  rho1 = deepcopy(rho0)
  beta = 1.0#0.9
  for i in range(ite-1):
     d1, err= projectE(rho1, fstate, known, normf, normr)
     #d2 = projectG(d1, support)
     #neg = np.where(d1<0)
     rho2 = deepcopy(d1)
     rho2[support] = rho1[support] - beta*d1[support]
     #rho2[neg] = rho1[neg] - beta*d1[neg]
     trace[i, 0:2]  = rho2[0:2]
     trace[i, 2] = err
     rho1 = deepcopy(rho2)
  rho2 = projectG(rho2, support, normr)
  rho2, err= projectE(rho2, fstate, known, normf, normr)
  trace[i+1, 0:2]  = rho2[0:2]
  return trace


# Gaussian set, for prior
def gauss(xdim, centerG, sigma):
  mask = np.zeros([xdim, xdim])
  norm0 = (1.0/(2.0*sigma*np.power(2*np.pi, 0.5)))
  norm1 = 2*sigma^2
  for i in range(xdim):
   xx2 = 1.0*np.power(i-centerG[0],2)
   for j in range(xdim):
     yy2 = 1.0*np.power(j-centerG[1],2)
     rr = xx2+yy2
     mask[i,j] = norm0*np.exp(-rr/norm1)
  return mask


# Gaussian Projection
def GaussP(rho, centerG, maskG, a1, b1, a2, b2):
   x, y = rho[0], rho[1]
   xx, yy = int(a1*x+b1), int(a2*y+b2)
   xp, yp = centerG[0]-xx, centerG[1]-yy 
   rhop = 0*rho
   rhop[0], rhop[1] = xp, yp
   return rhop


# Adaptive phase retrieval
def APR(ite, rho0, support, fstate, known, normf, normr, maskG, centerG, a1, b1, a2, b2):
  trace = np.zeros([ite,3]) # track the iterate, and error
  rho1 = deepcopy(rho0)
  rhop = 0*rho1
  d1, d2, epsilon, ermin = 1000,  0, 0.0001, 1000
  count, count2, beta0, beta = 0, 0, 0.05, 0
  db = 1.0*beta0/ite
  while (np.sum(np.abs(d1-d2))>epsilon and count <ite-1):
    d1 = projectG(rho1, support, normr)
    d2, err = projectE(2*d1-rho1, fstate, known, normf, normr)
    if err<ermin:
       ermin = err
    if (count >0 and err > 0.001):
     rhop = GaussP(rho1, centerG, maskG, a1, b1, a2, b2)
     beta = beta0-count2*db
     count2+=1
    else:
     beta = 0
    rho2 = rho1+d2-d1+beta*rhop
    trace[count, 0:2]  = rho2[0:2]
    trace[count, 2] = err
    rho1 = deepcopy(rho2)
    count+=1
  rho2 = projectG(rho2, support, normr)
  rho2, err= projectE(rho2, fstate, known, normf, normr)
  rho2 = projectG(rho2, support, normr)
  trace[count, 0:2]  = rho2[0:2]
  x1, y1 = convertcoord(rho2[0] ,rho2[1], a1, b1, a2, b2)
  print ('Last APR coord (%d, %d) with err %0.5f' %(x1, y1, err)) 
  return trace

def quickplot(fign, array1, array2, window, levels, gray=0):
  a1, a2 = window[0], window[1]
  fig = plt.figure(fign)
  fig.set_size_inches(9,9)
  ax = fig.add_subplot(111)
  plt.imshow(array1+array2, interpolation = 'nearest')
  ax.set_xlim(a1, a2)
  ax.set_ylim(a1, a2)
  ax.set_axis_off()
  if gray == 1:
     plt.gray()
  CS = plt.contour(array1, levels, colors = ('y', 'r', 'c'), linestyles='dashed', linewidths = 1.5)
  return ax

def iteplot(ax, i, ite, y0, x0, y1, x1, color, maptype, marker):
    if i==0:
       ax.plot(y0, x0, marker) # starting point
    if i==ite-1: # last iteration
       ax.plot(y1, x1, marker, label = maptype)
       ax.legend()
    if color == 'y':
       scale = 1.0
    if color == 'r':
       scale = 1.05
    if color == 'b':
       scale = 1.1
    plt.quiver(y0, x0, y1-y0, x1-x0, scale_units='xy', angles='xy', scale=scale, width = 0.005, color = color)
    y0, x0 = y1, x1
    return y0, x0
