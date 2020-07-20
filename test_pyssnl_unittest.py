"""
file to test the python ssnl agaisnt accepted values from the matlab version
"""

import unittest
import numpy as np
from scipy.io import loadmat
from ssnl import * #SSNL

testVals = loadmat('testVals.mat',squeeze_me=True)
ssnlTestObj = SSNL()
ssnlTestObj.genEqns()
ssnlTestObj.genGrids(testVals['nPts_IN'],testVals['dt_IN'],testVals['nZ_IN'])
ssnlTestObj.genFields()
ssnlTestObj.propagate()

def relError(A,B):
    
    def mag(arr,roundBool):
        if roundBool:
            return np.floor(np.log10(np.amax(abs(arr))))
        else:
            return np.ceil(np.log10(np.amax(abs(arr))))

    if np.all(A-B == 0):
        return float("30")
    
    if np.iscomplexobj(A) and np.iscomplexobj(B):
        valMag = [max([mag(A.real,0),mag(B.real,0)]),0]
        diffMag = [mag(A.real-B.real,1),0]
        valMag[1] = max([mag(A.imag,0),mag(B.imag,0)])
        diffMag[1] = mag(A.imag-B.imag,1)
        return min([valMag[0]-diffMag[0],valMag[1]-diffMag[1]])
    elif (not np.iscomplexobj(A)) and (not np.iscomplexobj(B)):
        valMag = max([mag(A,0),mag(B,0)])
        diffMag = mag(A-B,1)
        return valMag-diffMag
    else:
        raise ValueError('A very specific bad thing happened.')

class TestSSNL(unittest.TestCase):
    
    
    def test_fwhm(self):
        self.assertEqual(fwhm(testVals['fwhm_IN']), \
                          testVals['fwhm_OUT'], \
                          msg='pyssnl fwhm does not match MATLAB')
        
    def test_fft(self):        
        pyVal = fft(testVals['fft_IN'])
        self.assertTrue(relError(pyVal,testVals['fft_OUT']) >= 15,
                        msg='pyssnl fft does not match MATLAB')
        
    def test_ifft(self):        
        pyVal = ifft(testVals['ifft_IN'])
        self.assertTrue(relError(pyVal,testVals['ifft_OUT']) >= 15,
                        msg='pyssnl ifft does not match MATLAB')
        
    def test_intenPeak(self):
        pyVal = ssnlTestObj.intenPeak(testVals['ePulse_IN'],testVals['mRad_IN'],testVals['tau_IN'])
        self.assertTrue(relError(pyVal,testVals['intenPeak_OUT']) >= 15,
                        msg='pyssnl intenPeak does not match MATLAB')
            
    def test_fieldPeak(self):
        pyVal = ssnlTestObj.fieldPeak(testVals['ePulse_IN'],testVals['mRad_IN'],testVals['tau_IN'],testVals['n_IN'])
        self.assertTrue(relError(pyVal,testVals['fieldPeak_OUT']) >= 15,
                        msg='pyssnl fieldPeak does not match MATLAB')
        
    def test_intenField(self):
        pyVal = ssnlTestObj.intenField(testVals['intenField_IN'],testVals['n_IN'])
        self.assertTrue(relError(pyVal,testVals['intenField_OUT']) >= 15,
                          msg='pyssnl intenField does not match MATLAB')
        
    def test_energyField(self):
        pyVal = ssnlTestObj.energyField(testVals['intenField_IN'],testVals['n_IN'],testVals['mRad_IN'])
        self.assertTrue(relError(pyVal,testVals['energyField_OUT']) >= 15,
                        msg='pyssnl energyField does not match MATLAB')
        
    def test_indexEqns(self):
        pyVal = ssnlTestObj.eqns['index'][0](testVals['lams_IN'])
        pyVal = np.append(pyVal,ssnlTestObj.eqns['index'][1](testVals['lams_IN']))
        pyVal = np.append(pyVal,ssnlTestObj.eqns['index'][2](testVals['lams_IN'],testVals['theta_IN']))
        pyVal = pyVal.reshape((3,3))
        err = np.array([0,0,0])
        for ii in range(3):
            err[ii] = relError(pyVal[ii,:],testVals['indexEqn_OUT'][ii,:])
        self.assertTrue(np.all(err >= 15),
                        msg='pyssnl .eqns[\'index\'] does not match MATLAB')
        
    def test_dk_Eqns(self):
        pyVal = ssnlTestObj.eqns['dk']
        self.assertTrue(relError(pyVal,testVals['dkEqn_OUT'][0,0]) >= 15,
                        msg='pyssnl .eqns[\'k\'][:,1] does not match MATLAB')
        
    def test_NL_Eqns(self):
        pyVal = ssnlTestObj.eqns['nonLin'][0](testVals['eFields_IN'][1,:],testVals['eFields_IN'][2,:])
        pyVal = np.append(pyVal,ssnlTestObj.eqns['nonLin'][1](testVals['eFields_IN'][0,:],testVals['eFields_IN'][2,:]))
        pyVal = np.append(pyVal,ssnlTestObj.eqns['nonLin'][2](testVals['eFields_IN'][0,:],testVals['eFields_IN'][1,:]))
        pyVal = pyVal.reshape((3,-1))
        err = np.array([0,0,0])
        for ii in range(3):
            err[ii] = relError(pyVal[ii,:],testVals['NLEqns_OUT'][ii,:])
        self.assertTrue(np.all(err >= 15),
                        msg='pyssnl .eqns[\'nonLin\'] does not match MATLAB')
        
    def test_phase_Eqn(self):
        pyVal = ssnlTestObj.eqns['phase'](testVals['phase_IN'][0],testVals['phase_IN'][1],
                                      testVals['phase_IN'][2],testVals['phase_IN'][3],
                                      testVals['lamList_IN'],testVals['lams_IN'][0])
        self.assertTrue(relError(pyVal,testVals['phaseEqn_OUT']) >= 15,
                          msg='pyssnl .eqns[\'phase\'] does not match MATLAB')
        
    def test_dz(self):
        self.assertTrue(relError(ssnlTestObj.grids['dz'],testVals['dz_OUT']) >= 15,
                          msg='pyssnl .grids[\'dz\'] does not match MATLAB')
        
    def test_dw(self):
        self.assertTrue(relError(ssnlTestObj.grids['dw'],testVals['dw_OUT']) >= 15,
                          msg='pyssnl .grids[\'dw\'] does not match MATLAB')
        
    def test_lambda_List(self):
        pyVal = np.around(ssnlTestObj.lists['lambda']-testVals['lambdaList_OUT'],30)
        self.assertTrue(np.all((pyVal==0)), msg='pyssnl .list[\'lambda\'] does not match MATLAB')
        
    def test_t_List(self):
        self.assertTrue(relError(ssnlTestObj.lists['t'],testVals['tList_OUT']) >= 15,
                        msg='pyssnl .list[\'t\'] does not match MATLAB')
        
    def test_omega_List(self):
        self.assertTrue(relError(ssnlTestObj.lists['omega'],testVals['omegaList_OUT']) >= 15,
                        msg='pyssnl .list[\'omega\'] does not match MATLAB')
        
    def test_dOmega_List(self):
        self.assertTrue(relError(ssnlTestObj.lists['dOmega'],testVals['dOmegaList_OUT']) >= 15,
                        msg='pyssnl .list[\'dOmega\'] does not match MATLAB')
        
    def test_k_List(self):
        self.assertTrue(relError(ssnlTestObj.lists['k'],testVals['kList_OUT']) >= 15,
                        msg='pyssnl .list[\'k\'] does not match MATLAB')
        
    def test_genFields(self):
        N = len(ssnlTestObj.props['lams'])
        pyVal = np.zeros((N,ssnlTestObj.grids['nPts']),dtype=complex)
        for ii in range(N):
            pyVal[ii,:] = ssnlTestObj.eField['time'][ii+1][0,:]
        err = np.array([0,0,0])
        for ii in range(3):
            err[ii] = relError(pyVal[ii,:],testVals['eFields_OUT'][ii,:])
        self.assertTrue(np.all(err >= 13),
                        msg='pyssnl .eField (initial) does not match MATLAB')
        
    def test_propagate(self):
        pyVal = np.zeros((2,ssnlTestObj.grids['nPts']),dtype=complex)
        pyVal[0,:] = ssnlTestObj.eField['time'][3][-1,:]
        pyVal[1,:] = ssnlTestObj.eField['freq'][3][-1,:]
        err = np.array([0,0])
        for ii in range(2):
            err[ii] = relError(pyVal[ii,:],testVals['prop_OUT'][ii,:])
        self.assertTrue(np.all(err >= 12),
                        msg='pyssnl .eField (after .propagate) does not match MATLAB')
        
if __name__ == '__main__':
    unittest.main()
