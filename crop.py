#!/usr/bin/env python
import sys
import h5py

filename = sys.argv[1]
if len(sys.argv) > 2:
    c = int(sys.argv[2])
else:
    c = 100

with h5py.File(filename, 'a') as f:
    real_space_images = f['real_space_final']
    N, ny,nx = real_space_images.shape
    if 'thumbnail' in f.keys():
        del f['thumbnail']
    f.create_dataset('thumbnail', (N,ny-2*c,nx-2*c), dtype=real_space_images.dtype)
    f['thumbnail'].attrs['axes'] = ['experiment_identifier:y:x']
    for i in range(N):
        f['thumbnail'][i] = real_space_images[i,c:-c,c:-c]
    
        
