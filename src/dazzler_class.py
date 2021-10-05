#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 14:19:55 2020

@author: jackhirschman
"""


import numpy as np
import sys
import os
import re
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import pylab
import h5py
from scipy.interpolate import interp1d
import csv
import pandas as pd


class Dazzler_Pulse_Shaper():  
    def __init__(self, position, width, hole_position, hole_width, hole_depth,
                 delay, sec_order, third_order, fourth_order, c = 299792458.0):
        '''
        Initializes all dazzler parameters.
        These include central wavelength and width of input field; hole position, width, and depth; and 
        dispersion factors (first, second, third, and fourth order)
        '''
        self.position = position 
        self.width = width
        self.hole_position = hole_position
        self.hole_depth = hole_depth
        self.hole_width = hole_width
        
        self.delay = delay
        self.sec_order = sec_order
        self.third_order = third_order
        self.fourth_order = fourth_order
        
        self.c = c
    
    def calculate_parameters(self):
        '''
        Function uses parameters to derive other required constants.
        '''
        omega0 = 2*np.pi*self.c/self.position
        chi0 = self.width/(2*self.position)
        del_omega0 = omega0*(chi0-chi0**3)
        
        omega1 = 2*np.pi*self.c/self.hole_position
        chi1 = self.hole_width/(2*self.hole_position)
        del_omega1 = omega1*(chi1-chi1**3)/2
        
        return omega0, chi0, del_omega0, omega1, chi1, del_omega1
    
    def calculate_amplitude_transfer_function(self, ang_freq_vector, components_return = False):
        '''
        This function takes an angular frequency vector and uses the class parameters to 
        calculate the amplitude transfer function. This ends up being a product of 
        e^-([(w-w0)/delw0]^6) * (1-k*e^-([(w-w0)/delw0]^2))
        '''
        #get parameters
        omega0, chi0, del_omega0, omega1, chi1, del_omega1 = self.calculate_parameters()
        #calculate f and g
        f =(np.exp(-((ang_freq_vector-omega0)/del_omega0)**6))
        g = 1 - self.hole_depth*(np.exp(-((ang_freq_vector-omega1)/del_omega1)**2))
        
        # return either f,g, and A or just A, where A = f*g
        if (components_return):
            return f, g, f*g
        return f*g
    
    def calculate_phase_transfer_function(self, ang_freq_vector):
        '''
        This function calculates the phase transfer function. 
        '''
        #get parameters (this may be redundant, considering passing parameters around differently in future)
        omega0, chi0, del_omega0, omega1, chi1, del_omega1 = self.calculate_parameters()
        #calculate freq difference
        omega_dif = ang_freq_vector-omega0
        #return h as shown in equations in text
        return -(self.delay*omega_dif + self.sec_order/2 * omega_dif**2 + self.third_order/6 * omega_dif**3 + 
                 self.fourth_order/24 * omega_dif**4)
    
    def calculate_full_transfer_function(self,ang_freq_vector, S_saved=0, a_saved=0, components_return=False):
        '''
        This function calculates the full transfer function which is the product of the 
        phase and amplitude transfer functons. 

        Parameters
        ----------
        ang_freq_vector : TYPE
            DESCRIPTION.
        S_saved : TYPE, optional
            DESCRIPTION. The default is 0.
        a_saved : TYPE, optional
            DESCRIPTION. The default is 0.
        components_return : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        f,g,A_dial = None,None, None
        
        #calculate A dial
        if (components_return):
            f,g,A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=True)
        else:
            A_dial = self.calculate_amplitude_transfer_function(ang_freq_vector, components_return=False)
            
        #calculate phi dial
        phi_dial = self.calculate_phase_transfer_function(ang_freq_vector)
        
        #calculate S
        S_full = A_dial*np.exp(1j*phi_dial) + a_saved*np.exp(1j*phi_dial)*S_saved
        
        
        if (components_return):
            return f,g,A_dial,phi_dial, S_full
        
        return S_full
    
    def shape_input_pulse(self, E_field_input, time_vector, sampling_rate,
                          S_saved=0, a_saved=0, components_return=False):
        '''
        This function takes the input electric field, time vector, and sampling rate and calls
        on other functions in the class in order to calculate output electric field
        '''
        #take FT of input electric field
        E_field_input_ft = np.fft.fft(E_field_input)
        
        #get frequency vector and angular frequency vector based on sampling rate
        freq_vector = np.fft.fftfreq(n=(E_field_input_ft.size), d=1/sampling_rate)
        ang_freq_vector = 2*np.pi*freq_vector
        
        #calculate transfer function
        f,g,A_dial,phi_dial, S_full = None, None, None, None, None
        if (components_return):
            f,g,A_dial,phi_dial, S_full = self.calculate_full_transfer_function(ang_freq_vector, S_saved, a_saved, components_return)
            components_dict = {"f":f,"g":g,"A_dial":A_dial,"phi_dial":phi_dial,"S_full":S_full}

        else :
            S_full = self.calculate_full_transfer_function(ang_freq_vector, S_saved, a_saved, components_return)
            components_dict = {"S_full":S_full}
            
        #calculate output frequency domain electric field
        E_field_output_ft = E_field_input_ft*S_full
        
        #calculate IFT of output field
        E_field_output = np.fft.ifft(E_field_output_ft)
        
        
        return E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict
    
    def calculate_spectrum_phase(self, field):
        '''
        THis function separates spectrum and phase, unwrapping phase
        '''
        phase = -1*np.arctan2(np.imag(field),np.real(field))
        phase = np.unwrap(phase)
        spectrum = (np.abs(field))**2
        return spectrum, phase
    
    def convert_to_wavelength(self,I_freq, phase_freq, freq_vec,c, wavelength_vec_limits):
        '''
        This function converts the frequency domain signal into a wavelength domain. 
        Defaults to a wavelength vector that goes from 200nm below to 200nm above the central 
        wavelength. This should be adjusted if width of distribution is very large. 

        Parameters
        ----------
        I_freq : TYPE
            DESCRIPTION.
        phase_freq : TYPE
            DESCRIPTION.
        freq_vec : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.
        wavelength_vec_range : TYPE, optional
            DESCRIPTION. The default is 200e-9.

        Returns
        -------
        None.

        '''

        wavelength_vector = np.linspace(wavelength_vec_limits[0],wavelength_vec_limits[1],num=len(freq_vec))
        I_freq_interp = interp1d(2*np.pi*freq_vec, I_freq)
        I_wavelength = (2*np.pi*c/(wavelength_vector**2))*I_freq_interp(2*np.pi*c/wavelength_vector)
        phase_freq_interp = interp1d(2*np.pi*freq_vec, phase_freq)
        phase_wavelength = phase_freq_interp(2*np.pi*c/wavelength_vector)


        return wavelength_vector, I_wavelength, phase_wavelength    
    
    def calculate_total_spectrum_phase(self,E_field_input, E_field_input_ft, E_field_output, E_field_output_ft, S_full,
                             time_vector, freq_vector, wavelength_vector_limits):
        # Input 
        intensity_input_td, phase_input_td = self.calculate_spectrum_phase(E_field_input) # Time domain (td)
        spectrum_input_fd, phase_input_fd = self.calculate_spectrum_phase(E_field_input_ft) # Frequency domain (fd)
        wavelength_vector, spectrum_input_wd, phase_input_wd = self.convert_to_wavelength(spectrum_input_fd,
                                                                                     phase_input_fd,freq_vector,
                                                                                     self.c,wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd)
        
        input_functions = {"time_vector": time_vector,
                           "freq_vector":freq_vector,
                            "wavelength_vector":wavelength_vector,
                            "intensity_input_td":intensity_input_td,
                            "phase_input_td":phase_input_td,
                            "spectrum_input_fd":spectrum_input_fd,
                            "phase_input_fd":phase_input_fd,
                            "spectrum_input_wd":spectrum_input_wd,
                            "phase_input_wd":phase_input_wd}
                
        
        # Transfer function 
        S_full_td = np.fft.ifft(S_full)
        intensity_transfer_td, phase_transfer_td = self.calculate_spectrum_phase(S_full_td) # Time domain (td)
        spectrum_transfer_fd, phase_transfer_fd = self.calculate_spectrum_phase(S_full) # Frequency domain (fd)
        wavelength_vector,spectrum_transfer_wd, phase_transfer_wd = self.convert_to_wavelength(spectrum_transfer_fd,phase_transfer_fd,
                                                                        freq_vector, self.c,
                                                                        wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd) 
        
        transfer_functions = {"time_vector": time_vector,"freq_vector": freq_vector,
                              "wavelength_vector": wavelength_vector, "intensity_transfer_td": intensity_transfer_td,
                              "phase_transfer_td": phase_transfer_td,
                              "spectrum_transfer_fd":spectrum_transfer_fd,
                              "phase_transfer_fd":phase_transfer_fd,
                              "spectrum_transfer_wd":spectrum_transfer_wd,
                              "phase_transfer_wd":phase_transfer_wd}
              
        # Output  
        intensity_output_td, phase_output_td = self.calculate_spectrum_phase(E_field_output) # Time domain (td)
        spectrum_output_fd, phase_output_fd = self.calculate_spectrum_phase(E_field_output_ft) # Frequency domain (fd)
        wavelength_vector, spectrum_output_wd, phase_output_wd = self.convert_to_wavelength(spectrum_output_fd,phase_output_fd,
                                                                        freq_vector, self.c,
                                                                        wavelength_vec_limits=wavelength_vector_limits) #Wavelength domain (wd)
        output_functions = {"time_vector": time_vector,
                            "freq_vector":freq_vector,
                            "wavelength_vector":wavelength_vector,
                            "intensity_output_td":intensity_output_td,
                            "phase_output_td":phase_output_td,
                            "spectrum_output_fd":spectrum_output_fd,
                            "phase_output_fd":phase_output_fd,
                            "spectrum_output_wd":spectrum_output_wd,
                            "phase_output_wd":phase_output_wd}
                
        
          
        
        return input_functions, transfer_functions, output_functions
    
def prepare_input_output_pair_ensemble(dazzler_parameters_file, input_field_file, input_time_vectors, num_E_fields, input_field_params, saveFilePath, saveFileName, savePlotPath = None, savePlots = False):
    '''
    This function takes the inputs (dazzler parameters and input pulse) and output (spectrum/intensity and phase)
    and packages the data for use in the ML model. labels = parameters; logits = 1D vector of output flattened and E 
    field flattended
    '''
    #This may cause issues if too many lines
    dazzler_parameters = [];
    with open(dazzler_parameters_file, newline='') as fp:
        lines = csv.reader(fp, delimiter='\t')
        for line in lines:
            dazzler_param_dic = {}
            dazzler_param_dic["lam1"] = float(line[0])
            dazzler_param_dic["del_lam1"] = float(line[1])
            dazzler_param_dic["k"] = float(line[2])
            dazzler_param_dic["a1"] = float(line[3])
            dazzler_param_dic["a2"] = float(line[4])
            dazzler_param_dic["a3"] = float(line[5])
            dazzler_param_dic["a4"] = float(line[6])
            dazzler_parameters.append(dazzler_param_dic)
            
    E_field_inputs = np.loadtxt(input_field_file, usecols=range(0,num_E_fields), dtype = np.complex128)
    if(E_field_inputs.ndim==1):
        E_field_inputs=np.reshape(E_field_inputs,(-1,1))
    time_vectors = np.loadtxt(input_time_vectors, usecols=range(0, num_E_fields), dtype = np.float32)
    if(time_vectors.ndim==1):
        time_vectors=np.reshape(time_vectors,(-1,1))
        
    input_field_parameters = np.loadtxt(input_field_params, usecols=range(0, num_E_fields), dtype = np.float32)
    if(input_field_parameters.ndim==1):
        input_field_parameters=np.reshape(input_field_parameters,(-1,1))
    
    hierarchicalFileName = saveFilePath + saveFileName + ".hdf5"
    hierarchicalFile = h5py.File(hierarchicalFileName, "w")
    
    grp1 = hierarchicalFile.create_group("Runs")
            
    runNum = 0
    pulseLocal = None
    for jj in range(len(dazzler_parameters)):
        for ii in range(E_field_inputs.shape[1]):
            daz_param = dazzler_parameters[jj]
            daz_param["lam0"] = input_field_parameters[0,ii]
            daz_param["del_lam0"] = input_field_parameters[1,ii]
            if (pulseLocal):
                #pulseLocal.position=input_field_parameters[0,ii]
                #pulseLocal.width= input_field_parameters[1,ii]
                pulseLocal.position=daz_param["lam0"]
                pulseLocal.width= daz_param["del_lam0"]
                pulseLocal.hole_position= daz_param["lam1"]
                pulseLocal.hole_width= daz_param["del_lam1"]
                pulseLocal.hole_depth=daz_param["k"]
                pulseLocal.delay=daz_param["a1"]
                pulseLocal.sec_order=daz_param["a2"]
                pulseLocal.third_order=daz_param["a3"]
                pulseLocal.fourth_order=daz_param["a4"]
            else:
                pulseLocal = Dazzler_Pulse_Shaper(position=daz_param["lam0"],width= daz_param["del_lam0"],
                                           hole_position= daz_param["lam1"],hole_width= daz_param["del_lam1"],
                                           hole_depth=daz_param["k"], delay=daz_param["a1"], sec_order=daz_param["a2"],
                                           third_order=daz_param["a3"],fourth_order=daz_param["a4"])

            daz_param["lam0"] = pulseLocal.position
            daz_param["del_lam0"] = pulseLocal.width
            time_vector = time_vectors[:,ii]
            sample_rate = 1/(time_vector[1]-time_vector[0])
            E_field_input, E_field_input_ft, E_field_output, E_field_output_ft,time_vector, freq_vector, components_dict = pulseLocal.shape_input_pulse(E_field_inputs[:,ii],
                                                                                                                                                       time_vector, 
                                                                                                                                                       sample_rate, 
                                                                                                                                                       S_saved=0,
                                                                                                                                                       a_saved=0,
                                                                                                                                                       components_return=False)
            wavelength_vector_limits=[pulseLocal.position-250e-9, pulseLocal.position+250e-9]
            input_functions, transfer_functions, output_functions = pulseLocal.calculate_total_spectrum_phase(E_field_input, E_field_input_ft,
                                                                                                              E_field_output, E_field_output_ft,
                                                                                                              components_dict["S_full"], time_vector,
                                                                                                              freq_vector, wavelength_vector_limits=wavelength_vector_limits)
            #Format and save data
            groupName = "run"+str(runNum)
            grpNew = grp1.create_group(groupName)
            daz_params = grpNew.create_dataset("DazzlerParams",data=str(daz_param))
            
            grpNew.create_dataset("time_vector",data=input_functions['time_vector'])
            grpNew.create_dataset("freq_vector",data=input_functions['freq_vector'])
            grpNew.create_dataset("wavelength_vector",data=input_functions['wavelength_vector'])

            grpNew.create_dataset("intensity_input_td",data=input_functions['intensity_input_td'])
            grpNew.create_dataset("phase_input_td",data=input_functions['phase_input_td'])
            grpNew.create_dataset("spectrum_input_fd",data=input_functions['spectrum_input_fd'])
            grpNew.create_dataset("phase_input_fd",data=input_functions['phase_input_fd'])
            grpNew.create_dataset("spectrum_input_wd",data=input_functions['spectrum_input_wd'])
            grpNew.create_dataset("phase_input_wd",data=input_functions['phase_input_wd'])
            
            grpNew.create_dataset("intensity_output_td",data=output_functions['intensity_output_td'])
            grpNew.create_dataset("phase_output_td",data=output_functions['phase_output_td'])
            grpNew.create_dataset("spectrum_output_fd",data=output_functions['spectrum_output_fd'])
            grpNew.create_dataset("phase_output_fd",data=output_functions['phase_output_fd'])
            grpNew.create_dataset("spectrum_output_wd",data=output_functions['spectrum_output_wd'])
            grpNew.create_dataset("phase_output_wd",data=output_functions['phase_output_wd'])
            
            grpNew.create_dataset("intensity_transfer_td",data=transfer_functions['intensity_transfer_td'])
            grpNew.create_dataset("phase_transfer_td",data=transfer_functions['phase_transfer_td'])
            grpNew.create_dataset("spectrum_transfer_fd",data=transfer_functions['spectrum_transfer_fd'])
            grpNew.create_dataset("phase_transfer_fd",data=transfer_functions['phase_transfer_fd'])
            grpNew.create_dataset("spectrum_transfer_wd",data=transfer_functions['spectrum_transfer_wd'])
            grpNew.create_dataset("phase_transfer_wd",data=transfer_functions['phase_transfer_wd'])

            runNum += 1
             
    
            # Plot and Save
            if (savePlots):
                print("Do later")
    hierarchicalFile.close()
    # Shape waveform, no plotting, retrieve fields in specified domains(s)
    
    return 
