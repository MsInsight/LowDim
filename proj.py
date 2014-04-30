#! /usr/bin/env python
# By Sara Salha  
# Aprile 2014

'''This code compares the performance of AP, DM and APR '''


import numpy as np                    
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from sys import argv, exit
from copy import deepcopy
from lowdim import *


if len(argv) < 6:
    print './proj.py  flg0, flg1, flg2, flg3'
    print 'flg0 0/1 random start point: 0 new, 1 old'
    print 'flg00 0/1 Gaussian prior location: 0 new, 1 old'
    print 'flg1, flg2: stepy, stepx (offset of prior)'
    print 'flg3, save flag'
    exit(1)

flg0 = int(argv[1])
flg00 = int(argv[2])
flg1 = int(argv[3]) 
flg2 = int(argv[4]) 
flg3 = int(argv[5])
print '\n'

# parameters
#--------------
a, b = 30, 301 # Landscape size/resolution
ite = 70 # iteration number
d0 = 9  # oversample dimensions (make sure odd, as this is a 3pixel sys)
sigma = 10
uncert = 0.2



if flg0 ==0:
  rhoin = np.random.uniform(-a/2.5,a/2.5, d0) # random start
  np.savez('rhoin.npz', rhoin = rhoin) 
if flg0 ==1:
  rhoin = np.load('rhoin.npz')['rhoin']

# Original density
#-----------------
a0, b0, c0 = 2, 7, 1 # density points
rho_0 = np.zeros([d0])
rho_0[0], rho_0[1], rho_0[2] = a0, b0, c0
normr = np.sum(rho_0)
support = np.where(rho_0==0)


# Fourier transform
#-------------------
modData0 =( np.fft.fft( rho_0 ))
fstate = np.abs( modData0)
known = np.where(fstate>0)
normf = np.sum(np.power(fstate,2))

# Energy landscape
#-------------------
mask =  plotellipsoids(a, b, normr, fstate) # set intersection
ydim, xdim = np.shape(mask)
window = [0, 1*xdim] #[0, xdim]
landscape = energy_landscape(a, b, d0, normr, normf, fstate, known) 
np.savez('landscape.npz', landscape=landscape)

#plt.figure(1)
#plt.imshow(np.log(landscape+1.0),interpolation = 'nearest')
#plt.imshow(np.log(landscape+1.0)+0.1*mask,interpolation = 'nearest')
#plt.show()
#exit(1)

# convert coord
#--------------
kk0 = (b-1)/2
kk = np.where(landscape ==np.min(landscape))
aa1, bb1, aa2, bb2 = computeshift(kk, rho_0, kk0)
x0, y0 = convertcoord(rhoin[0], rhoin[1], aa1, bb1, aa2,bb2) # starting point
xga, yga = convertcoord(rho_0[0], rho_0[1], aa1, bb1, aa2, bb2) # global min
xgb, ygb = convertcoord(rho_0[2], rho_0[1], aa1, bb1, aa2, bb2) # global min (twin)
# prior location
#---------------
if flg00 ==0:
  epsilong1 = np.random.uniform(-uncert*xga, uncert*xga, 1)
  epsilong2 = np.random.uniform(-uncert*yga, uncert*yga,1)
  centerG = [int(xga+epsilong1), int(yga+epsilong2)]
  np.savez('centerG.npz', centerG = centerG)
if flg00 ==1:
  centerG = np.load('centerG.npz')['centerG']
  centerG+=[flg1,flg2]

traceAP  =  AP(ite, rhoin, support, fstate, known, normf, normr, aa1, bb1, aa2, bb2)
traceDM  =  DM(ite, rhoin, support, fstate, known, normf, normr, aa1, bb1, aa2, bb2)
#tracehio =  hio(ite, rhoin, support, fstate, known, normf, normr)
maskG = gauss(xdim, centerG, sigma) # Gaussian mask
traceAPR =  APR(ite, rhoin,support, fstate, known, normf, normr, maskG, centerG, aa1, bb1, aa2, bb2)


array1 = np.log(np.log(landscape+1.0)+0.1)
ma, me, mi = np.max(array1), np.mean(array1), np.min(array1)
levels = [mi+28*me,  mi+15*me, mi+2.88*me]
array2 = 0.1*mask+100*maskG
ax = quickplot(1, array1, array2, window, levels, gray=1)
# Plot Global min
for t1 in np.arange(-1, 1, 1):
  for t2 in np.arange(-1, 1, 1):
    ax.plot(yga+t1, xga+t2, 'cs') 
    ax.plot(ygb+t1, xgb+t2, 'cs') 

# Initialize
yap0, xap0 = y0, x0
ydm0, xdm0 = y0, x0
yapr0, xapr0 = y0, x0

# Iterate
for i in range(ite):
    # AP
    x1, y1 = traceAP[i,0], traceAP[i,1]
    xap1, yap1 = convertcoord(x1, y1, aa1, bb1, aa2, bb2) 
    yap0, xap0 =  iteplot(ax, i, ite, yap0, xap0, yap1, xap1, 'y', r'$\mathrm{AP}$', 'yo')
    
    # DM
    x1, y1 = traceDM[i,0], traceDM[i,1]
    if (y1!=0 and x1!=0):
       xdm1, ydm1 = convertcoord(x1, y1, aa1, bb1, aa2, bb2) 
    ydm0, xdm0 =  iteplot(ax, i, ite, ydm0, xdm0, ydm1, xdm1, 'r', r'$\mathrm{DM}$', 'ro')
    
    # APR
    x1, y1 = traceAPR[i,0], traceAPR[i,1]
    if (y1!=0 and x1!=0):
       erap = traceAPR[i,2]
       xapr1, yapr1 = convertcoord(x1, y1, aa1, bb1, aa2, bb2) 
    yapr0, xapr0 =  iteplot(ax, i, ite, yapr0, xapr0, yapr1, xapr1, 'b', r'$\mathrm{APR}$', 'bo')
    
plt.title(r'$\mathrm{Low\ dimensional\ iterative\ methods}$')
name = 'Lowdim%03d'%(flg3)+'.png'
#if erap <1:
plt.savefig(name)

plt.show()
