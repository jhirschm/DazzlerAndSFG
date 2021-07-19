import pyssnl_amy_2 as ssnl_amy
import sympy as sp
import matplotlib.pyplot as plt
from scipy.io import loadmat
import efnc as efnc
import pickle as pickle 
import numpy as np

u = ssnl_amy.UNITS()

#unpickling file input from Rose: 1030-330-NoTransferOutput 'OutputforAmy_EFieldandTime_notransferfunc.txt'
with open('OutputforAmy_SamplingRate200_noshape.txt', 'rb') as handle:
    eField1= handle.read()
input_eField = pickle.loads(eField1)

plt.figure(1)
plt.title('Dazzler: Intensity of input E field + E field')
plt.plot(input_eField['E_field'])
plt.xlim(7900,8500)
plt.plot(abs(np.array(input_eField['E_field']))**2)
plt.show() #just close the pop-up window and allow the code to run 

#functions that I will use later 
from contextlib import nullcontext
import numpy as np
import sympy as sp
from numpy.fft import fftshift, ifftshift


def fft(field):
    '''fft with shift
    
    Shifting values so that initial time is 0.
    Then perform FFT, then shift 0 back to center.
    
    field: 1xN numpy array
    
    return a 1xN numpy array'''
    return fftshift(np.fft.fft(ifftshift(field)))

def ifft(field):
    '''ifft with shift
        
    Shifting values so that initial time is 0.
    Then perform IFFT, then shift 0 back to center.
    
    field: 1xN numpy array
        
    return a 1xN numpy array'''
    return fftshift(np.fft.ifft(ifftshift(field)))

#define taylor expansion phase stuff
c = 299792458
eqns = {'index':None, 'dk':None, 'nonLin':None, 'phase':None}

dOmega = lambda l, lCtr:(2*np.pi*c) * ( (1/lCtr) - (1.0/l) )
eqns['phase'] = lambda kk2, kk3, kk4, kk5, l, lCtr:(
            ( (kk2/np.math.factorial(2)) * (dOmega(l,lCtr)**2) ) +
            ( (kk3/np.math.factorial(3)) * (dOmega(l,lCtr)**3) ) +
            ( (kk4/np.math.factorial(4)) * (dOmega(l,lCtr)**4) ) +
            ( (kk5/np.math.factorial(5)) * (dOmega(l,lCtr)**5) )
            )
tay12 = -0.5*2.2
tay13 = 1.05*-0.1
specPhases = np.array([
                          [-tay12*u.ps**2,tay13*u.ps**3,0,0], 
                          [tay12*u.ps**2,-tay13*u.ps**3,0,0],
                          [0,0,0,0]
                          ])


#generating lambda vectors 

gridKeys = ['nPts','dt','dz','nZ','dw']
listKeys = ['t','lambda','omega','dOmega','k']
        
grids = {key:None for key in gridKeys}
lists = {key:None for key in listKeys}
        
nFields = 2
grids['nPts'] = len(input_eField['E_field']) #nPts
grids['dt'] = (input_eField['time_vector'][2] - input_eField['time_vector'][1]) #for my vector don't use u.fs --> * u.fs #dt in fs(?)
grids['nZ'] = 100
grids['dw'] = (2*np.pi) / (grids['nPts'] * grids['dt']) #EQUAL TO (self.input_pk['freq_vector'][2] - self.input_pk['freq_vector'][1])*(2*np.pi)
        
#lists['t'] = grids['dt'] * (np.arange(-grids['nPts']/2,grids['nPts']/2)+1)#self.input_eField['time_vector'] # EQUAL TO self.grids['dt'] * (np.arange(-self.grids['nPts']/2 ,self.grids['nPts']/2))
lists['dOmega'] = grids['dw'] * (np.arange(-grids['nPts']/2,grids['nPts']/2)+1)#self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2))# np.sort(self.input_eField['freq_vector'])*2*np.pi*(1/u.fs)#for my vector don't use u.fs*(1/u.fs)# equal to self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2))
lists['lambda'] = np.zeros((1,grids['nPts']))
lists['omega'] = np.zeros((1,grids['nPts']))


ks = (2*np.pi)/(1030*u.nm)
omegas   = c * ks  #m/s x 1/m = 1/s 

lists['omega'] = lists['dOmega'] + omegas
lists['lambda'] = np.divide(2*np.pi*c,lists['omega'])

#Actual electric field calculations

nFields = 2
for ii in range(nFields): 
    nZ = 100 #number of z steps in crystal 
    nPts = len(input_eField['E_field'])  

    timeField  = {(ii+1):
                    np.zeros((nZ+1,nPts),dtype=complex) 
                    for ii in range(nFields)
                    }
    freqField  = {(ii+1):
                    np.zeros((nZ+1,nPts),dtype=complex) 
                    for ii in range(nFields)
                    }
    eField  = {'time':timeField, 'freq':freqField}

    #the assigning of the electric field to the input dazzler value
    eField['time'][ii+1][0,:] =input_eField['E_field']#self.input_eField['E_field']#np.sqrt(abs(np.array(self.input_eField['E_field'])**2)) 

    import matplotlib.pyplot as plt
    print(ii)
    plt.figure(ii+1)  
    plt.title('no spec phases')        
    plt.plot(abs(eField['time'][ii+1][0,:])**2)

    #go to frequency space
    eField['freq'][ii+1][0,:] = fft(eField['time'][ii+1][0,:])


    #apply stretching and compressing phases                     
    eField['freq'][ii+1][0,:] *= (
    np.exp( 1j * eqns['phase'](specPhases[ii,0],
                                                    specPhases[ii,1],
                                                    specPhases[ii,2],
                                                    specPhases[ii,3],
                                                    lists['lambda'],
                                                    1030*u.nm
                                                    )
                        )
                    )
    eField['time'][ii+1][0,:] = ifft(eField['freq'][ii+1][0,:])

            #plotting test
    plt.title('after specPhases')        
    plt.plot(abs(eField['time'][ii+1][0,:])**2)
plt.show()  