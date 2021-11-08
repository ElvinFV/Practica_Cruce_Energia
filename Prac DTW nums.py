# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 00:10:11 2021

@author: adria
"""

from scipy.io import wavfile
from matplotlib import pyplot as plt
from winsound import *
from mpl_toolkits.mplot3d import Axes3D

import wave
import numpy as np


import librosa
import librosa.display

def dp(dist_mat):
    """
    Find minimum-cost path through matrix `dist_mat` using dynamic programming.

    The cost of a path is defined as the sum of the matrix entries on that
    path. See the following for details of the algorithm:

    - http://en.wikipedia.org/wiki/Dynamic_time_warping
    - https://www.ee.columbia.edu/~dpwe/resources/matlab/dtw/dp.m

    The notation in the first reference was followed, while Dan Ellis's code
    (second reference) was used to check for correctness. Returns a list of
    path indices and the cost matrix.
    """

    N, M = dist_mat.shape
    
    # Initialize the cost matrix
    cost_mat = np.zeros((N + 1, M + 1))
    for i in range(1, N + 1):
        cost_mat[i, 0] = np.inf
    for i in range(1, M + 1):
        cost_mat[0, i] = np.inf

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            penalty = [
                cost_mat[i, j],      # match (0)
                cost_mat[i, j + 1],  # insertion (1)
                cost_mat[i + 1, j]]  # deletion (2)
            i_penalty = np.argmin(penalty)
            cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
            traceback_mat[i, j] = i_penalty

    # Traceback from bottom right
    i = N - 1
    j = M - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i = i - 1
            j = j - 1
        elif tb_type == 1:
            # Insertion
            i = i - 1
        elif tb_type == 2:
            # Deletion
            j = j - 1
        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return (path[::-1], cost_mat)

plt.close('all')

filename = '0_1.wav'
y, sr = librosa.load(filename)
Persona, _ = librosa.effects.trim(y)
librosa.display.waveplot(Persona, sr=sr);

AA=Persona
senal1N=AA/np.max(AA)
#ventanas de 1ms
#400 datos
Bin1 = np.where(np.abs(senal1N) >= 0.1, 1, 0)
senal1 = []
for i in range (0, len(Bin1)-440, 1):
    senal1.append(np.mean( Bin1[i : i+440] ))

plt.plot(senal1)

#Se retira todas las partes muertas
#se elimina el ruido y solo nos quedamos
Bin2 = np.where(np.array(senal1) >= 0.1, 1, 0)
senal2 = []
for i in range(0, len(Bin2)-439, 1):
    if (Bin2[i] == 1):
        senal2.append(senal1N[i])

plt.figure(3)
plt.plot(senal2)
plt.show()

#==============Filtro de preenfasis

corre = np.zeros(len(senal2))
corre[1:-1] = senal2[0:-2]
pre = corre - (0.95*np.array(senal2))

plt.figure(4)
plt.plot(pre)
plt.show()

for i in range(0, len(pre), 1):
    if (np.abs(np.float16(pre[i])) >= 0.9):
        pre[i] = 0
#=============Ventanado de Fourier
frame = 400#Tamaño ventana
overlap = 80#Desplazamiento
Ima = []
k = 0
for i in range (0, len(pre)-(frame-1), overlap):
    k = k + 1
    Fou = np.abs(np.fft.fft( pre[i:frame-1 + i] * np.hamming(frame-1) ))
    Ima.append(Fou[0:220])


#X Tiempo
#Y Espectro de Furier
#Z Magnitud
X,Y =np.mgrid[0:k, 0:220]
Z=np.array(Ima)
fig = plt.figure(6)
ax=fig.gca(projection='3d')
ax.set_xlabel('Tiempo')
ax.set_ylabel('Fourier')
ax.set_zlabel('Magnitud')
surf=ax.plot_surface(X,Y,Z, cmap= 'coolwarm', linewidth=0)

#Espectro inconfundible de la misma palabra,no importa su velocidad
comp1 = np.max(np.array(Ima), axis = 0)
comp2 = np.mean(np.array(Ima), axis = 0)
Patron = (comp1 + comp2)/2
plt.figure(7)
plt.plot(Patron)

Patron=Patron/np.max(Patron)
def DTW(A,B):
    x=A#Vector 1
    y=B##Vector 2
    N = x.shape[0]
    M = y.shape[0]
    dist_mat = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            dist_mat[i, j] = abs(x[i] - y[j])
    
    # DTW1
    path, cost_mat = dp(dist_mat)
    Costo=cost_mat[N - 1, M - 1]
    return Costo

# np.savetxt('9_Patron.csv', Patron)
L0 = np.loadtxt('0_Patron2.csv')
L1 = np.loadtxt('1_Patron.csv')
L2 = np.loadtxt('2_Patron.csv')
L3 = np.loadtxt('3_Patron.csv')
L4 = np.loadtxt('4_Patron.csv')
L5 = np.loadtxt('5_Patron.csv')
L6 = np.loadtxt('6_Patron.csv')
L7 = np.loadtxt('7_Patron.csv')
L8 = np.loadtxt('8_Patron.csv')
L9 = np.loadtxt('9_Patron.csv')

dtw_0=DTW(Patron,L0)
dtw_1=DTW(Patron,L1)
dtw_2=DTW(Patron,L2)
dtw_3=DTW(Patron,L3)
dtw_4=DTW(Patron,L4)
dtw_5=DTW(Patron,L5)
dtw_6=DTW(Patron,L6)
dtw_7=DTW(Patron,L7)
dtw_8=DTW(Patron,L8)
dtw_9=DTW(Patron,L9)

Vals=[dtw_0,dtw_1,dtw_2,dtw_3,dtw_4,dtw_5,dtw_6,dtw_7,dtw_8,dtw_9]
if dtw_0==min(Vals):
    print("El número reconocido es cero")
if dtw_1==min(Vals):
    print("El número reconocido es uno")  
if dtw_2==min(Vals):
    print("El número reconocido es dos")   
if dtw_3==min(Vals):
    print("El número reconocido es tres")   
if dtw_4==min(Vals):
    print("El número reconocido es cuatro")    
if dtw_5==min(Vals):
    print("El número reconocido es cinco")    
if dtw_6==min(Vals):
    print("El número reconocido es seis")
if dtw_7==min(Vals):
    print("El número reconocido es siete")  
if dtw_8==min(Vals):
    print("El número reconocido es ocho")   
if dtw_9==min(Vals):
    print("El número reconocido es nueve")   

    