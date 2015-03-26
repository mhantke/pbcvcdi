import spimage,numpy
import h5py
from matplotlib.pyplot import figure,imsave

def full_reconstruction(filename,index_number,S,N_rec=100,downsampling=2):
    I,M = load_pattern(filename,downsampling)
    O_N = reconstruct(I,M,S,N_rec,"./%s_rec.cxi" % index_number)
    ferr = O_N["scores_final"]["fourier_error"]
    rs = O_N["real_space_final"]
    sup = O_N["support_final"]
    sel = ferr[:] < S["fourier_error_max"]
    p = prtf(rs[sel,:,:],sup[sel,:,:],"./prtf_%s.h5" % index_number)
    plot_prtf(p,"./%s" % (index_number),downsampling)

def load_pattern(filename,downsampling=2):
    img = spimage.sp_image_read(filename,0)
    I = abs(img.image)
    M = numpy.array(img.mask,dtype="bool")
    spimage.sp_image_free(img)
    I,M = spimage.downsample(I,downsampling,mask=M)
    return I,M

def reconstruct(I,M,S,N=None,filename=None):
    R = spimage.Reconstructor()
    R.set_intensities(I,shifted=False)
    R.set_mask(M,shifted=False)
    R.set_number_of_iterations(4000)
    R.set_initial_support(radius=10)
    R.set_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=2000)
    R.append_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=1000)
    R.append_phasing_algorithm("er",constraints=["enforce_positivity"],number_of_iterations=1000)
    R.set_support_algorithm("area",blur_init=6.,blur_final=6.,area_init=S["area_init_1st"],area_final=S["area_final_1st"],update_period=100,number_of_iterations=2000,center_image=False)
    R.append_support_algorithm("area",blur_init=6.,blur_final=1.,area_init=S["area_init_2nd"],area_final=S["area_final_2nd"],update_period=100,number_of_iterations=1000,center_image=False)
    R.append_support_algorithm("static",number_of_iterations=1000,center_image=False)
    if N is None:
        R.set_number_of_outputs(100)
        O = R.reconstruct()
    else:
        R.set_number_of_outputs(3)
        O = R.reconstruct_loop(N)
        if filename is not None:
            print filename
            W = spimage.CXIWriter(filename,N)
            W.write_stack(O)
            W.close()
    return O

def prtf(real_space,support,filename=None):
    prtf = spimage.prtf(real_space,support,
                        translate=True,enantio=True,full_out=True)
    if filename is not None:
        with h5py.File(filename,"w") as f:
            for k,v in prtf.items():
                f[k] = v
    return prtf

def plot_prtf(prtf,filename_root,downsampling):
    P = spimage.downsample(prtf["prtf"],downsampling)
    Pr = spimage.radial_mean(P,cx=P.shape[1]/2,cy=P.shape[0]/2)
    from scipy.ndimage.filters import gaussian_filter
    Prs = gaussian_filter(Pr, 4)
    x = numpy.arange(len(Pr)) + 0.5
    binning = 8 * downsampling
    pixel_size = binning*75E-6
    detector_distance = 0.73
    wavelength = 1.1E-9
    res = spimage.detector_pixel_to_resolution_element(i_pixel=x,pixel_size=pixel_size,detector_distance=detector_distance,wavelength=wavelength)
    fig = figure(figsize=(5,3))
    q = 1/res/2.*1E-9
    if (Prs<1/numpy.e).sum() > 0:
        q_res = (q[Prs<1/numpy.e])[0]
    else:
        q_res = q[-1]
    ax = fig.add_subplot(111)
    ax.plot(q,Pr)
    ax.plot(q,Prs)
    ax.axhline(1/numpy.e,ls="--",color="black")
    ax.axvline(q_res,ls="--",color="black")
    ax.text(q_res+0.0025,0.8,"R = %.1f nm" % (1/q_res),ha="left")
    ax.set_xlabel("Resolution shell [1/nm]")
    ax.set_ylabel("PRTF")
    ax.set_ylim(0,1)
    #title("Fourier_error_max:%s" % (S["fourier_error_max"]))
    fig.savefig(filename_root+"_prtf.png",dpi=300)
    I = abs(prtf["super_image"])
    imsave(filename_root+"_super.png",I)
    from python_tools import imgtools,gentools,colormaptools
    Ic = imgtools.crop(I,40,"center_of_mass")
    imsave(filename_root+"_super_cropped.png",Ic)
    Icu = imgtools.upsample(Ic,20)
    imsave(filename_root+"_super_cropped_upsampled.png",Icu,cmap=colormaptools.cmaps["jet_lightbg2"])
