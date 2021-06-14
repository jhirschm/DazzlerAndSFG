"""
build test file, should not appear in git
"""

import ssnl
# import numpy as np
# import sympy as sp
import matplotlib.pyplot as plt
# from scipy.io import loadmat

# testVals = loadmat('testVals.mat',squeeze_me=True)
a = ssnl.SSNL()
u = ssnl.UNITS()

gdd = 22.50
sf = 0.8

a.set_default(gdd*1000,sf)

a.genEqns()

a.genGrids()

a.genFields()

a.propagate()

# SIMPLE END PROP PLOT
plt.figure(99)
py = abs(a.eField['time'][3][-1,:])**2
plt.plot(py)
plt.title('Final Field')
plt.xlim(7500,9000)


# # NONLIN EQNS DIFF
# plt.figure(1)
# fieldMap = np.array([[2,3],[1,3],[1,2]])
# py = np.zeros((a.grids['nPts'],),dtype=complex)
# mat = np.zeros((a.grids['nPts'],),dtype=complex)
# py = a.eqns['nonLin'][2](
#                 a.eField['time'][fieldMap[2,0]][0,:],
#                 a.eField['time'][fieldMap[2,1]][0,:]
#                 )
# mat = testVals['NLEqns_OUT'][2,:]
# plt.plot(py.real-mat.real)
# plt.plot(py.imag-mat.imag)
# plt.title('Nonlinear Equation Difference')
# plt.savefig('nonLin_diff')

# # PHASE EQN DIFF
# plt.figure(2)
# py = a.eqns['phase'](testVals['phase_IN'][0],testVals['phase_IN'][1],
#                                       testVals['phase_IN'][2],testVals['phase_IN'][3],
#                                       testVals['lamList_IN'],testVals['lams_IN'][0])
# mat = testVals['phaseEqn_OUT']
# plt.plot(py-mat)
# plt.title('Phase Equation Difference')
# plt.savefig('phase_diff')
    
# # INITIAL FIELD DIFF
# plt.figure(3)
# for ii in range(2):
#     py = abs(a.eField['time'][ii+1][0,:])**2
#     mat = abs(testVals['eFields_OUT'][ii,:])**2
#     plt.plot(py-mat)
#     # plt.plot(np.divide((py-mat),mat))
# plt.title('Initial Field Difference')
# plt.savefig('field_diff')

# # fft EQN DIFF
# plt.figure(4)
# py = a.fft(testVals['fft_IN'])
# mat = testVals['fft_OUT']
# plt.plot(py.real-mat.real)
# plt.plot(py.imag-mat.imag)
# plt.title('FFT Difference')
# plt.legend(['real','imaginary'],loc=3)
# plt.savefig('fft_diff')

# # ifft EQN DIFF
# plt.figure(5)
# py = a.ifft(testVals['ifft_IN'])
# mat = testVals['ifft_OUT']
# plt.plot(py.real-mat.real)
# plt.plot(py.imag-mat.imag)
# plt.title('IFFT Difference')
# plt.legend(['real','imaginary'],loc=3)
# plt.savefig('ifft_diff')

# # k List DIFF
# plt.figure(6)
# py = a.lists['k'][2,:]
# mat = testVals['kList_OUT'][2,:]
# plt.plot(py-mat)
# plt.title('IFFT Difference')
# plt.legend(['1','2','3'],loc=3)
# plt.savefig('k_lists_diff')

# # lambda List DIFF
# plt.figure(7)
# py = a.lists['lambda'][2,:]
# mat = testVals['lambdaList_OUT'][2,:]
# plt.plot(py-mat)
# plt.title('IFFT Difference')
# plt.legend(['1','2','3'],loc=3)
# plt.savefig('k_lists_diff')
    
# # INDEX DIFF
# py = a.eqns['index'][0](testVals['lams_IN'])
# py = np.append(py,a.eqns['index'][1](testVals['lams_IN']))
# py = np.append(py,a.eqns['index'][2](testVals['lams_IN'],testVals['theta_IN']))
# py = py.reshape((3,3))
# mat = testVals['indexEqn_OUT']
# print(py-mat)

# # FIELD PEAK DIFF
# py = a.fieldPeak(a.props['energies'][0],
#                  a.props['spotRad'],
#                  a.props['taus'][0],
#                  a.eqns['index'][0](a.props['lams'][0])
#                  )
# py = np.append(py,a.fieldPeak(a.props['energies'][0],
#                  a.props['spotRad'],
#                  a.props['taus'][0],
#                  testVals['n_IN']Nah 
#                  )
#                )
# mat = testVals['fieldPeak_OUT']
# print(py-mat)

# tmp = a.eField['time'][3][-1,:]
# plt.figure(99)
# plt.plot(abs(tmp)**2)
# plt.xlim(7500,9000)

# z = -1
# field = 3
# py = a.eField['time'][field][z,:]
# mat = testVals['totalField'][0,z,field-1,:]
# plt.figure(98)
# plt.plot(abs(py)**2-abs(mat)**2)
# plt.xlim(7250,9000)



        