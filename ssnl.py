import numpy as np
import sympy as sp
from numpy.fft import fftshift, ifftshift


def gdd(dt,dt0):
    '''calculate GDD'''
    #http://toolbox.lightcon.com/tools/pulsebroadening/
    sqrt = np.sqrt( (dt/dt0)**2 -1 )
    gdd  = (dt0**2)/(4*np.log(2)) * sqrt
    return gdd

def fwhm(lst):
    '''Calculate the fwhm given list of intensity values
    
    lst: 1xN list or numpy array
    
    returns a single value as the number of indicies !!NOT SCALED!!'''
    if np.iscomplexobj(lst):
        lst  = abs(lst)**2
    highVals = np.where(lst > np.amax(lst)/2)
    fwhm     = highVals[0][-1]-highVals[0][0]+1
    return fwhm

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
    
    def __init__(self):
        
        u = UNITS()
        self.c          = 299792458 * (u.m/u.s)
        self.eps0       = (8.854187817 * 10**-12) / u.m
        self.w0_2_fwhm  = 4 * np.log(2)
        self.lams       = None
        self.ks         = None
        self.omegas     = None
        self.crys       = None
        self.len        = None
        self.theta      = None
        self.mixType    = None
        self.taus       = None
        self.energies   = None
        self.spotRad    = None
        self.specPhases = None


    def set_default(self, newfwhm, sf):
        '''Set properties to case with:
        sf      = scale factor btwn tay 12, tay 13 
        newfwhm = pulse length in ps for green 
        1030 nm = fundamental wavelength 
        515  nm = second harmonic
        330  fs = dt0, in gdd formula
        
        results in squarish pulse '''
        u     = UNITS() 
        dt0   = 246 #330 #fs
        tay12 = gdd(newfwhm, dt0)*10**-6
        tay13 = (tay12/8)*sf
           
        self.lams     = np.array([1024*u.nm,1024*u.nm,512*u.nm])
            #1030*u.nm,1030*u.nm,515*u.nm])
        self.ks       = (2*np.pi)/self.lams
        self.omegas   = self.c * self.ks
        self.crys     = 'BBO'
        self.len      = 0.5*u.mm
        self.theta    = 23.29
        self.mixType  = 'SFG'
        self.taus     = np.array([246*u.fs,246*u.fs,20*u.fs])
            #330*u.fs,330*u.fs,20*u.fs])
        self.energies = np.array([25*u.uJ,25*u.uJ,0*u.uJ])
        self.spotRad  = 400*u.um
        self.specPhases = np.array([ 
                          [-tay12*u.ps**2,tay13*u.ps**3,0,0], 
                          [tay12*u.ps**2,-tay13*u.ps**3,0,0],
                          [0,0,0,0]])
        return    
                          
    def intenPeak(self,ePulse, mRad, tau):
        '''Calculate peak intensity, assumes Gausian
        
        ePulse: 1x1 float (total energy in field)
        mRad: 1x1 float (1/e^2 width of the beam)
        tau: 1x1 float (fwhm of pulse in time)
        
        returns a single value'''
        # Look at mathematica notebook for derivation
        numerator   = 2*ePulse*np.sqrt(np.log(16))
        denominator = (np.pi**(3/2))*(mRad**2)*tau 
        return numerator/denominator
    
    
    def fieldPeak(self,ePulse, mRad, tau, n):
        '''Calculate peak amplitude of E-Field, assumes Gausian
        
        ePulse: 1x1 float (total energy in field)
        mRad: 1x1 float (1/e^2 width of the beam)
        tau: 1x1 float (fwhm of pulse in time)
        n: 1x1 float (index of refraction)
    
        returns a single value.'''
        # Look at mathematica notebook for derivation
        aFieldPeak = np.sqrt( self.intenPeak(ePulse,mRad,tau) / (2*self.eps0*n*self.c) );
        return aFieldPeak
    
    
    ## --- Put these into a class? ----
    def intenField(self,field,n):
        '''Converts E-field to intesity of a single time slice of the field.
        
        field: 1xN numpy array (E-field)
        n: 1x1 float (index of refraction)
        
        returns an array same size as entered.'''
        # Look at mathematica notebook for derivation
        return np.real( (2*self.eps0*n*self.c) * np.conj(field)*field )
    
    def energyField(self, field,n,mRad):
        '''Calculate total energy of a single time slice of the field
        
        field: 1xN numpy array (E-field)
        n: 1x1 float (index of refraction)
        mRad: 1x1 float (radius of spot size)
        
        returns a single value.'''
        # Look at mathematica notebook for derivation
        const = (np.pi/2)*mRad**2
        return const * np.sum(self.intenField(field,n)) * self.grids['dt']
        
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
        
        self.grids['nPts'] = nPts
        self.grids['dt'] = dt
        self.grids['nZ'] = nZ
        self.grids['dz'] = self.len / (self.grids['nZ'] - 1)
        self.grids['dw'] = (2*np.pi) / (self.grids['nPts'] * self.grids['dt'])
        
        self.lists['t'] = self.grids['dt'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1)
        self.lists['dOmega'] = self.grids['dw'] * (np.arange(-self.grids['nPts']/2,self.grids['nPts']/2)+1)
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
                
                self.eField['time'][ii+1][0,:] = (
                    self.fieldPeak(self.energies[ii],
                                   self.spotRad,
                                   self.taus[ii],
                                   self.eqns['index'][ii](self.lams[ii])
                                   ) *
                    np.exp( -(1/2) * self.w0_2_fwhm *
                           ( (self.lists['t'] / self.taus[ii])**2)
                           )
                    )
                
            elif ii == nFields-1:
            
                self.eField['time'][ii+1][0,:] = (
                    self.fieldPeak(self.energies[ii],
                                   self.spotRad,
                                   self.taus[ii],
                                   self.eqns['index'][ii](
                                       self.lams[ii],self.theta
                                       )
                                   ) *
                    np.exp( -(1/2) * self.w0_2_fwhm *
                           ( (self.lists['t'] / self.taus[ii])**2)
                           )
                    )
            
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
