'''
|==================================== crpa_beamstear_visual.py ====================================|
|                                                                                                  |
|   Property of Daniel Sturdivant. Unauthorized copying of this file via any medium would be       |
|   super sad and unfortunate for me. Proprietary and confidential.                                |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     scripts/crpa_beamstear_visual.py                                                     |
|   @brief    Basic CRPA visualization tools.                                                      |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib import cm


#* ANTENNA MODEL
SPEED_OF_LIGHT = 299792458.0              #! [m/s]
FREQUENCY = 1575.42e6                     #! [Hz]
WAVELENGTH = SPEED_OF_LIGHT / FREQUENCY   #! [m]
R2D = 180 / np.pi
D2R = np.pi / 180

# N_ANT = 7                                 #! number of elements
# RADIAL_DIST = WAVELENGTH/2                #! [m] linear separation
# DEG_SEP = 360/6                           #! [deg] angular separation
# N_ANT = 4                                 #! number of elements
# RADIAL_DIST = WAVELENGTH/2                #! [m] linear separation
# DEG_SEP = 360/4                           #! [deg] angular separation
N_ANT = 3                                #! number of elements
RADIAL_DIST = WAVELENGTH/2                #! [m] linear separation
DEG_SEP = 360/3                          #! [deg] angular separation

AZ = 45.0 * D2R
EL = 0.0 * D2R
UNIT = np.array([np.cos(AZ)*np.cos(EL), np.sin(AZ)*np.cos(EL), np.sin(EL)])

#--------------------------------------------------------------------------------------------------#
def plot_3d_array_factor(ant_xyz, weights, title):
  
  el_span_b = np.linspace(0, np.pi/2, 180)
  az_span_b = np.linspace(0, 2*np.pi, 720)
  
  pattern_3d = np.zeros((180*720, 3))
  i,j,q = 0,0,0
  for el in el_span_b:
    j = 0
    for az in az_span_b:
      u = np.array([np.sin(az)*np.cos(el), np.cos(az)*np.cos(el), np.sin(el)])
      p = 0
      for k in range(N_ANT):
        p += weights[k]*np.exp(1j * 2 * np.pi * (ant_xyz[:,k] @ u) / WAVELENGTH)
      # gain_pattern[j,i] = 10*np.log10(np.abs(p))
      gain = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
      pattern_3d[q,:] = np.linalg.norm(p) * gain
      j += 1
      q += 1
    i += 1
    
  norm_gain = np.linalg.norm(pattern_3d, axis=1)
  f = plt.figure()
  ax = f.add_subplot(111) #, projection='3d')
  im = ax.scatter(pattern_3d[:,0], pattern_3d[:,1], pattern_3d[:,2], c=norm_gain, cmap='gnuplot') # cmap='turbo')
  f.colorbar(im, ax=ax)
  ax.grid(visible=True, which='both')
  ax.set_xlabel('X (boresight)')
  ax.set_ylabel('Y (horizontal)')
  # ax.set_zlabel('Z (vertical)')
  f.suptitle(f'{title}, Gain Pattern (Amplitude) of Panel In Cartesian Coordinates')
  
  return f, ax
    
#--------------------------------------------------------------------------------------------------#


if __name__ == '__main__':
  
  # initialize antenna positions
  ant_xyz = np.zeros((3, N_ANT))
  if N_ANT >= 7 and N_ANT % 2 != 0:
    for i in range(1, N_ANT):
      angle = (i-1) * DEG_SEP * D2R
      ant_xyz[0,i] = RADIAL_DIST * np.cos(angle)
      ant_xyz[1,i] = RADIAL_DIST * np.sin(angle)
  else:
    for i in range(N_ANT):
      angle = (i) * DEG_SEP * D2R
      ant_xyz[0,i] = RADIAL_DIST * np.cos(angle)
      ant_xyz[1,i] = RADIAL_DIST * np.sin(angle)
  
  # BS antenna
  bs_weights = np.zeros(N_ANT, dtype=complex)
  for i in range(N_ANT):
    spatial_phase = 2 * np.pi * (ant_xyz[:,i] @ UNIT) / WAVELENGTH
    bs_weights[i] = np.exp(-1j * spatial_phase)
  
  # NS antenna (scale factors)
  sf = np.ones(N_ANT)
  sf[1:] = -1 / (N_ANT - 1)
  ns_weights = np.zeros(N_ANT, dtype=complex)
  for i in range(N_ANT):
    spatial_phase = 2 * np.pi * (ant_xyz[:,i] @ UNIT) / WAVELENGTH
    ns_weights[i] = np.exp(-1j * spatial_phase) * sf[i]
    
    
  f0, ax0 = plt.subplots()
  ax0.scatter(ant_xyz[0,:], ant_xyz[1,:], marker='*')
  ax0.set_xlabel('East [m]')
  ax0.set_ylabel('North [m]')
  f1, ax1 = plot_3d_array_factor(ant_xyz, bs_weights, 'Beam Stearing')
  f2, ax2 = plot_3d_array_factor(ant_xyz, ns_weights, 'Null Stearing')
  plt.show()