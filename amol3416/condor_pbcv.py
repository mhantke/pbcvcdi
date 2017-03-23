import numpy

try:
    import matplotlib.pyplot as pypl
    plotting = True
except:
    plotting = False

import os,time
this_dir = os.path.dirname(os.path.realpath(__file__))

import condor

import logging
logger = logging.getLogger("condor")
#logger.setLevel("DEBUG")
#logger.setLevel("WARNING")
logger.setLevel("INFO")

N = 1
rotation_formalism="random"
rotation_values = None

# Source
src = condor.Source(wavelength=1.59E-9, pulse_energy=1E-3, focus_diameter=1E-6)
# Detector
det = condor.Detector(distance=0.37, pixel_size=0.0003, nx=256, ny=256)
# Map
#print "Simulating map"
par = condor.ParticleMap(diameter=None, material_type="poliovirus", geometry="custom",
                         emd_id="5384",
                         rotation_formalism=rotation_formalism, rotation_values=rotation_values)
s = "particle_map"
E = condor.Experiment(src, {s : par}, det)

W = condor.utils.cxiwriter.CXIWriter("./5384.cxi")
for i in range(N):
    t = time.time()
    res = E.propagate()
    #print time.time()-t
    if plotting:
        real_space = numpy.fft.fftshift(numpy.fft.ifftn(res["entry_1"]["data_1"]["data_fourier"]))
        pypl.imsave(this_dir + "/simple_test_%s_%i.png" % (s,i), numpy.log10(res["entry_1"]["data_1"]["data"]))
        pypl.imsave(this_dir + "/simple_test_%s_%i_rs.png" % (s,i), abs(real_space))
    W.write(res)
W.close()
