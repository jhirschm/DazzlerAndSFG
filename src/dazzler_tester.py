import numpy as np
import dazzler_class
import matplotlib.pyplot as plt

trial = dazzler_class.Dazzler_Pulse_Shaper(1030e-9, 330e-15, 1030e-9, 4e-9,1,0,0,0,0)
time_vector, EofT = trial.make_gaussian_pulse()
E_field_input, E_field_input_ft, E_field_output, E_field_output_ft, time_vector, freq_vector, components_dict=trial.shape_input_pulse(EofT, time_vector)
