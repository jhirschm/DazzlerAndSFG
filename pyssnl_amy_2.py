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

class UNITS:
    
    def __init__(self,mScale=0,sScale=0):
        
        self.m = 10**mScale
        self.mm = 10**(-3*self.m)
        self.um = 10**(-6*self.m)
        self.nm = 10**(-9*self.m)
        
        self.s = 10**sScale
        self.ns = 10**(-9*self.s)
        self.ps = 10**(-12*self.s)
        self.fs = 10**(-15*self.s)
        
        self.J = (self.m**2)/(self.s**2)
        self.mJ = 10**(-3*self.J)
        self.uJ = 10**(-6*self.J)

class SSNL:
    
    def __init__(self, input_eField):

        #vector peak input
        self.input_eField = input_eField
        
        #constants and units
        u = UNITS()
        self.c          = 299792458 * (u.m/u.s)
        self.eps0       = (8.854187817 * 10**-12) / u.m
        self.w0_2_fwhm  = 4 * np.log(2)
        
        #maybe later - assign all of these using the input_pk dictionary? i
        self.lams       = None
        self.ks         = None
        self.omegas     = None
        self.crys       = None
        self.length     = None
        self.theta      = None
        self.mixType    = None
        self.taus       = None
        self.energies   = None
        self.spotRad    = None
        self.specPhases = None


    def set_default(self, newfwhm=22.5*1000, sf=0.8, specphase_1 =1.05*-0.1, specphase_2 = -0.5*2.2 ):
        '''Set properties to case with:
        sf      = scale factor btwn tay 12, tay 13 
        newfwhm = pulse length in ps for green 
        1030 nm = fundamental wavelength 
        515  nm = second harmonic
        330  fs = dt0, in gdd formula        
        results in squarish pulse '''

        u     = UNITS() 

    ##Initial scaling parameters 
        
        tay13 = specphase_1
        tay12 = specphase_2
    ##Input vairables 

        #wavelengths, frequency, and wavevector(const) of two incoming beams + SFG beam
        #self.inten_max_indx = np.argmax(self.input_pk['intensity_output_td'])
        #self.crt_lambda = self.input_pk['wavelength_vector'][self.inten_max_indx]
        laser_wavelength = 1030 #was 1030
        self.lams     = np.array([laser_wavelength*u.nm, laser_wavelength*u.nm, laser_wavelength*0.5*u.nm])#np.array([self.crt_lambda*u.nm,self.crt_lambda*u.nm,self.crt_lambda*u.nm]) #[1024*u.nm,1024*u.nm,512*u.nm])
        self.ks       = (2*np.pi)/self.lams
        self.omegas   = self.c * self.ks  #m/s x 1/m = 1/s 

    #time widths, pulse energies, spectral phases, physical width of gaussian
        #calculate taus from input gaussian 
        #self.high_inten_vals = np.where(self.input_pk['intensity_output_td']> 0.1*self.input_pk['intensity_output_td'][self.inten_max_indx])
        #self.high_inten_time = [self.input_pk['time_vector'][indx] for indx in self.high_inten_vals]
        #assumes input is in fs
        #self.tau_1 = self.high_inten_time[0][len(self.high_inten_time[0])-1] - self.high_inten_time[0][0] 
        self.taus     = [246*u.fs,246*u.fs,20*u.fs] #np.array([self.tau_1*u.fs,self.tau_1*u.fs,0]) #[246*u.fs,246*u.fs,20*u.fs]330*u.fs,330*u.fs,20*u.fs])
        self.energies = np.array([17*u.uJ,17*u.uJ,0*u.uJ])   #[used to be 25 uJ]  

    #stretching and compressing spectral phases    
        self.specPhases = np.array([
                          [-tay12*u.ps**2,tay13*u.ps**3,0,0], 
                          [tay12*u.ps**2,-tay13*u.ps**3,0,0],
                          [0,0,0,0]
                          ])

        self.spotRad  = 1000*u.um #used to be 400

    ##Crystal System variables 
        self.crys     = 'BBO'
        self.length      = 2*u.mm #0.5 mm
        self.theta    = 23.29
        self.mixType  = 'SFG'
           
        return    


    def genEqns(self,crysName=None):
        '''Creates the anonymous functions for the index of refraction, nonlinear mixing,
        taylor expansion of phase, and the derivative of 'k' for the speed of the grids.
        
        crysName: (OPTIONAL) a string. Name of the nonlinear crystal to use.
        DOES NOT WORK RIGHT NOW AS BBO IS THE ONLY INCLUDED ONE
        
        returns nothing but sets internal attributes
        '''
        if crysName is None: # Future support for other crystals
            crysName = self.crys
            
        u = UNITS()
        (l, theta, w, lCtr, field1, field2, field3, dOmega, kk2, kk3, kk4, kk5)\
            = sp.symbols('l theta w lCtr field1 field2 field3 dOmega kk2 kk3 kk4 kk5')
        
        # if crysName == 'BBO' # Future support for other crystals
        
        nO_SYMPY = sp.sqrt( 2.7359 + 0.01878/((l/u.um)**2 - 0.01822) - 0.01354 * (l/u.um)**2 )
        nO = lambda l:np.sqrt( 2.7359 + 0.01878/((l/u.um)**2 - 0.01822) - 0.01354 * (l/u.um)**2 )
        nE = lambda l:np.sqrt( 2.3753 + 0.01224/((l/u.um)**2 - 0.01667) - 0.01516 * (l/u.um)**2 )
        dNL = 2.01 * 10**-12
        
        nE_Theta = lambda l, theta:np.sqrt( 1 / (
            np.cos(np.deg2rad(theta))**2/nO(l)**2 +
            np.sin(np.deg2rad(theta))**2/nE(l)**2
            ))
        
        self.eqns = {'index':None, 'dk':None, 'nonLin':None, 'phase':None}
                
        self.eqns['index'] = np.array((nO,nO,nE_Theta))
        
        k1 = (w/self.c)*nO_SYMPY.subs(l,(2*np.pi*self.c)/w)
        dk1 = sp.diff(k1,w)
        self.eqns['dk'] = float(dk1.subs(w,self.omegas[0]).evalf())
        
        nonLinCoef = (((dNL * 1j) * 2 * self.ks[0])/self.eqns['index'][0](self.lams[0]),
                      ((dNL * 1j) * 2 * self.ks[1])/self.eqns['index'][1](self.lams[1]),
                      ((dNL * 1j) * 2 * self.ks[2])/self.eqns['index'][2](self.lams[2],self.theta),
                      )
        
        self.eqns['nonLin'] = np.array(((lambda field2, field3: nonLinCoef[0] * np.conj(field2) * field3),
                               (lambda field1, field3: nonLinCoef[1] * np.conj(field1) * field3),
                               (lambda field1, field2: nonLinCoef[2] * field1 * field2),
                               ))
        
        dOmega = lambda l, lCtr:(2*np.pi*self.c) * ( (1/lCtr) - (1.0/l) )
        self.eqns['phase'] = lambda kk2, kk3, kk4, kk5, l, lCtr:(
            ( (kk2/np.math.factorial(2)) * (dOmega(l,lCtr)**2) ) +
            ( (kk3/np.math.factorial(3)) * (dOmega(l,lCtr)**3) ) +
            ( (kk4/np.math.factorial(4)) * (dOmega(l,lCtr)**4) ) +
            ( (kk5/np.math.factorial(5)) * (dOmega(l,lCtr)**5) )
            )
        
        pass

            
    def genGrids(self,nPts=2**14,dt=None,nZ=100):
        '''Creates the .grids and .lists attributes of the object for the run.
        The .grids attribute holds the info for the discrete step spacing of
        quantities such as time and space. The .lists property holds all the
        points used in computation based on the spacing from .grids and
        values from self.properties
        
        nPts: (OPTIONAL) a single integer. Number of points in lists. 
                You will regret everything if it is not apower of 2.
                DEFAULT: 2**14
        dt: (OPTIONAL) a single integer. The spacing in time but also defines
                the frequency resolution. More time, tighter resolution of nPts
                around the central frequencies 
                DEFAULT: tau[1]/10
        nZ: (OPTIONAL) a single integer. Number of steps to take through the
                simulation. Higher numbers result in more accurate simulations
                but take more time linearly
                DEFAULT: 100
                
        returns nothing but sets internal attributes
        '''
        if dt is None:
            dt = self.taus[0]/10
        
        gridKeys = ['nPts','dt','dz','nZ','dw']
        listKeys = ['t','lambda','omega','dOmega','k']
        
        self.grids = {key:None for key in gridKeys}
        self.lists = {key:None for key in listKeys}
        
        nFields = len(self.lams)
        
        #modify
        u = UNITS()
        self.grids['nPts'] = len(self.input_eField['time_vector']) #nPts
        self.grids['dt'] = (self.input_eField['time_vector'][2] - self.input_eField['time_vector'][1]) #for my vector don't use u.fs --> * u.fs #dt in fs(?)
        self.grids['nZ'] = nZ
        self.grids['dz'] = self.length / (self.grids['nZ'] - 1)
        self.grids['dw'] = (2*np.pi) / (self.grids['nPts'] * self.grids['dt']) #EQUAL TO (self.input_pk['freq_vector'][2] - self.input_pk['freq_vector'][1])*(2*np.pi)
        
        self.lists['t'] = self.grids['dt'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1)#self.input_eField['time_vector'] # EQUAL TO self.grids['dt'] * (np.arange(-self.grids['nPts']/2 ,self.grids['nPts']/2))
        self.lists['dOmega'] = self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1)#self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2))# np.sort(self.input_eField['freq_vector'])*2*np.pi*(1/u.fs)#for my vector don't use u.fs*(1/u.fs)# equal to self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2))
        self.lists['lambda'] = np.zeros((nFields,self.grids['nPts']))
        self.lists['omega'] = np.zeros((nFields,self.grids['nPts']))
        self.lists['k'] = np.zeros((nFields,self.grids['nPts']))
        

        for ii in range(nFields):
            
            self.lists['omega'][ii,:] = self.lists['dOmega'] + self.omegas[ii]
            self.lists['lambda'][ii,:] = np.divide(2*np.pi*self.c,self.lists['omega'][ii,:])
            
            if ii != nFields-1:
                self.lists['k'][ii,:] = (
                    np.divide(2*np.pi,self.lists['lambda'][ii,:]) *
                     self.eqns['index'][ii](self.lists['lambda'][ii,:]) - 
                      (self.lists['dOmega']*self.eqns['dk']
                       )
                     )
            elif ii == nFields-1:
                self.lists['k'][ii,:] = (
                    np.divide(2*np.pi,self.lists['lambda'][ii,:]) *
                     self.eqns['index'][ii](self.lists['lambda'][ii,:],self.theta) - 
                      (self.lists['dOmega']*self.eqns['dk']
                       )
                     )
        
        return
    
    
    
    def genFields(self):
        '''Creates the field variables and allocates memory. This is based on all
        the attributes input and generated before hand.
        
        returns nothing but sets internal attributes        
        '''
        
        nFields = len(self.lams)
        
        timeField  = {(ii+1):
                   np.zeros((self.grids['nZ']+1,self.grids['nPts']),dtype=complex) 
                   for ii in range(nFields)
                   }
        freqField  = {(ii+1):
                   np.zeros((self.grids['nZ']+1,self.grids['nPts']),dtype=complex) 
                   for ii in range(nFields)
                   }
            
        self.eField  = {'time':timeField, 'freq':freqField}
        
        for ii in range(nFields):
            
            if ii != nFields-1:
                n_indx = self.eqns['index'][ii](self.lams[ii])
                self.eField['time'][ii+1][0,:] =np.sqrt( abs(np.array(self.input_eField['E_field'])**2) / (2*self.eps0*n_indx*self.c) )#self.input_eField['E_field']#np.sqrt(abs(np.array(self.input_eField['E_field'])**2)) 
                
            elif ii == nFields-1:
            
                self.eField['time'][ii+1][0,:] = np.zeros(len(self.input_eField['time_vector']))

            import matplotlib.pyplot as plt
            plt.figure(ii+1)  
            plt.title('no spec phases')        
            plt.plot(abs(self.eField['time'][ii+1][0,:])**2)
            if ii == 0:
                self.before = {'b4': None, '5eva' : None}
                self.before['b4'] = np.exp( 1j * self.eqns['phase'](self.specPhases[ii,0],
                                                self.specPhases[ii,1],
                                                self.specPhases[ii,2],
                                                self.specPhases[ii,3],
                                                self.lists['lambda'][ii,:],
                                                self.lams[ii]
                                                )
                       )
                self.before['5eva'] = self.input_eField['E_field']
                       #abs(self.eField['time'][ii+1][0,:])**2

            self.eField['freq'][ii+1][0,:] = fft(self.eField['time'][ii+1][0,:])
                      
            self.eField['freq'][ii+1][0,:] *= (
                np.exp( 1j * self.eqns['phase'](self.specPhases[ii,0],
                                                self.specPhases[ii,1],
                                                self.specPhases[ii,2],
                                                self.specPhases[ii,3],
                                                self.lists['lambda'][ii,:],
                                                self.lams[ii]
                                                )
                       )
                )
            
            self.eField['time'][ii+1][0,:] = ifft(self.eField['freq'][ii+1][0,:])

            #plotting test
            plt.title('after specPhases')        
            plt.plot(abs(self.eField['time'][ii+1][0,:])**2)            
            # if ii == 0: 
                # plt.figure(99)
                # plt.plot(np.exp( 1j * self.eqns['phase'](self.specPhases[ii,0],
                #                                      self.specPhases[ii,1],
                #                                      self.specPhases[ii,2],
                #                                      self.specPhases[ii,3],
                #                                      self.lists['lambda'][ii,:],
                #                                      self.lams[ii]
                #                                      )))
        plt.show()             
        return
    
    def RKstep(self, zStep):
        '''Custom Runga-Kutta 4 algorithm to work with class structure.
        
        zStep: a single integer. Index of which step in propagation we are on.
                Is used to index the .efield property so between 1 and .grids['nZ']
        
        returns nothing but sets internal attributes
        '''
        
        nFields = len(self.lams)
        N = self.grids['nPts']
        
        if nFields == 3:
            fieldMap = np.array([[2,3],[1,3],[1,2]])
            
        rk0 = np.zeros((nFields,N),dtype=complex)
        rk1 = np.zeros((nFields,N),dtype=complex)
        rk2 = np.zeros((nFields,N),dtype=complex)
        rk3 = np.zeros((nFields,N),dtype=complex)
        
        for ii in range(nFields):
            rk0[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:],
                self.eField['time'][fieldMap[ii,1]][zStep,:]
                )
            
        for ii in range(nFields):
            
            rk1[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk0[fieldMap[ii,0]-1,:]/2,
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk0[fieldMap[ii,1]-1,:]/2
                )
            
        for ii in range(nFields):
            rk2[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk1[fieldMap[ii,0]-1,:]/2,
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk1[fieldMap[ii,1]-1,:]/2
                )
            
        for ii in range(nFields):
            rk3[ii,:] = self.grids['dz'] * self.eqns['nonLin'][ii](
                self.eField['time'][fieldMap[ii,0]][zStep,:] + rk2[fieldMap[ii,0]-1,:],
                self.eField['time'][fieldMap[ii,1]][zStep,:] + rk2[fieldMap[ii,1]-1,:]
                )
        
        for ii in range(nFields):
            self.eField['time'][ii+1][zStep,:] = (
                self.eField['time'][ii+1][zStep,:] +
                rk0[ii,:]/6 + rk1[ii,:]/3 + rk2[ii,:]/3 + rk3[ii,:]/6
                )
        
        
        return
    
    def propagate(self):
        '''Propagate the field along the crystal and update the .eField property.
        Each step and field is held in memory so the zStep that we are on is the
        first index in the .eField['time'] or .eField['time'] property 
        
        
        returns nothing but sets internal attributes
        '''
        
        def dzStep(self, iz):
            if iz == 1 or iz == self.grids['nZ']:
                return self.grids['dz']/2
            else:
                return self.grids['dz']
        
        nFields = len(self.lams)
        
        for iZ in range(1,self.grids['nZ']+1):
            
            for iF in range (nFields):
                self.eField['time'][iF+1][iZ,:] = ifft( 
                    self.eField['freq'][iF+1][iZ-1,:] *
                    np.exp(1j * self.lists['k'][iF,:] * dzStep(self,iZ))
                    )
                
            if iZ <= self.grids['nZ']-1:
                
                self.RKstep(iZ)
                
            for iF in range(nFields):
                self.eField['freq'][iF+1][iZ,:] = fft(
                    self.eField['time'][iF+1][iZ,:]
                    )
            
            
        return
