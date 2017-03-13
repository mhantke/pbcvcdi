#!/usr/bin/env python
import sys,os
import h5py
import spimage
import numpy as np

index = int(sys.argv[1])
base_path = "/Users/benedikt/data/LCLS/amol3416/pbcv/phasing/r0098/"
filename = base_path + "r0098_%03d_0.h5" %index
img = spimage.sp_image_read(filename,0)
I = abs(img.image)
M = np.array(img.mask,dtype="bool")
spimage.sp_image_free(img)

# Phasing parameters
niter_raar = 5000
niter_hio  = 0
niter_er   = 1000
niter_total = niter_raar + niter_hio + niter_er
beta = 0.9
support_radius = 11.0
support_area   = ((support_radius**2)*np.pi) / (I.shape[0]**2)

# Run phasing
R = spimage.Reconstructor()
R.set_intensities(I)
R.set_mask(M)
R.set_number_of_iterations(niter_total)
R.set_number_of_outputs_images(6)
R.set_number_of_outputs_scores(200)
R.set_initial_support(radius=support_radius)
#R.append_support_algorithm("static", number_of_iterations=niter_hio)
#R.append_support_algorithm("threshold", center_image=True, update_period=20,
#                       blur_init=5, blur_final=1, threshold_init=0.3, threshold_final=0.15)
R.append_support_algorithm("area", center_image=True, update_period=20, number_of_iterations=niter_raar+niter_er,
                                                  blur_init=3, blur_final=1, area_init=support_area, area_final=0.005)
#R.append_support_algorithm("static",  number_of_iterations=niter_er)
R.append_phasing_algorithm("raar",beta_init=beta, beta_final=beta, number_of_iterations=niter_raar,
                                                      constraints=['enforce_real', 'enforce_positivity'])
#R.append_phasing_algorithm("hio", beta_init=beta, beta_final=beta, number_of_iterations=niter_hio,
#                           constraints=['enforce_real', 'enforce_positivity'])
R.append_phasing_algorithm("er",  number_of_iterations=niter_er,
                                                      constraints=['enforce_real', 'enforce_positivity'])
output = R.reconstruct_loop(1000)
R.clear()

M = 50
N = 20
output_filename = base_path + "r0098_%03d_phased.h5" %index
os.system('rm %s' %(output_filename))
for n in range(N):
    output = R.reconstruct_loop(M)
    print "Done Reconstructions: %d/%d" %((n+1)*M, N*M)
    with h5py.File(output_filename, 'a') as f:
        for k,v in output.iteritems():
            if isinstance(v,dict):
                if k not in f.keys():
                    f.create_group(k)
                for kd,vd in v.iteritems():
                    if kd not in f[k].keys():
                        f[k].create_dataset(kd, (N*M,), dtype=vd.dtype)
                        d.attrs['axes'] = ['experiment_identifier']
                    f[k][kd][n*M:(n+1)*M] = vd
            else:
                if k not in f.keys():
                    f.create_dataset(k, (N*M, v.shape[1], v.shape[2]), dtype=v.dtype)
                    d.attrs['axes'] = ['experiment_identifier:y:x']
                f[k][n*M:(n+1)*M] = v
