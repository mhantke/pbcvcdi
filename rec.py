import spimage,numpy,os
import h5py
import copy
import mulpro
import multiprocessing
import h5writer
from matplotlib.pyplot import figure
from python_tools.gentools import imsave

mulpro_logger = mulpro.logger
mulpro_logger.setLevel("ERROR") 
mulpro.log.WARNING_AFTER_JOB_DURATION_SEC = 400

class MulproReconstructor(spimage.Reconstructor):
    def __init__(self, nrecons=1, ncpus=None, filename=None):
        spimage.Reconstructor.__init__(self)
        self.nrecons  = nrecons
        self.ncpus    = ncpus
        self.nperworker = 1
        self.filename = filename
        self.prepare_mulpro()
        
    def prepare_mulpro(self):
        self.counter = 0
        self.njobs = self.nrecons / self.nperworker
        if self.filename is None:
            self.filename = './tmp.cxi'
        if os.path.isfile(self.filename):
            print "INFO: %s already exists and is now overwritten." % self.filename
            os.system('rm %s' %self.filename)
        if self.ncpus is None:
            self.ncpus = multiprocessing.cpu_count()

    def recons_worker(self, iandN):
        i,N = iandN
        o = self.reconstruct_loop(N)
        res = {}
        res['i'] = i
        res['o'] = o
        return res
        
    def reader(self):
        if self.counter < self.njobs:
            self.counter += 1
            return [self.counter - 1, self.nperworker]
        else:
            return None
        
    def writer(self, res):
        i = res['i']
        output = res['o']
        with h5py.File(self.filename, 'a') as f:
            for k,v in output.iteritems():
                if isinstance(v,dict):
                    if k not in f.keys():
                        f.create_group(k)
                    for kd,vd in v.iteritems():
                        if kd not in f[k].keys():
                            d = f[k].create_dataset(kd, (self.nrecons,), dtype=vd.dtype)
                            d.attrs['axes'] = ['experiment_identifier']
                        f[k][kd][i*self.nperworker:(i+1)*self.nperworker] = vd
                else:
                    if k not in f.keys():
                        d = f.create_dataset(k, (self.nrecons, v.shape[1], v.shape[2]), dtype=v.dtype)
                        d.attrs['axes'] = ['experiment_identifier:y:x']
                    f[k][i*self.nperworker:(i+1)*self.nperworker] = v

    def reconstruct_mulpro_loop(self):
        mulpro.mulpro(Nprocesses=self.ncpus, worker=self.recons_worker, getwork=self.reader, logres=self.writer)

def full_reconstruction(filename,index_number,S):
    os.system("mkdir -p %s" % index_number)
    os.system("rm %s/*" % index_number)
    N_rec = int(S["n_rec"])
    downsampling = int(S["downsampling"])
    I,M = load_pattern(filename,index_number,downsampling)
    O_N = reconstruct(I,M,S,N_rec,index_number)
    rs = O_N["real_space_final"]
    ferr = O_N["scores_final"]["fourier_error"]
    rerr = O_N["scores_final"]["real_error"]
    imsave_reconstructions(rs,ferr,index_number)
    plot_real_error(rerr,index_number)
    ferr_thresh = float(S["fourier_error_max"])
    plot_fourier_error(ferr,index_number,ferr_thresh)

    sup = O_N["support_final"]
    sel = ferr[:] < ferr_thresh
    if sel.sum() > 1:
        p = prtf(rs[sel,:,:],sup[sel,:,:],index_number)
        plot_prtf(p,index_number,downsampling)
    else:
        print "WARNING: Did not find more than one reconstruction below threshold."

def load_pattern(filename,index_number,downsampling=2):
    img = spimage.sp_image_read(filename,0)
    I = abs(img.image)
    M = numpy.array(img.mask,dtype="bool")
    I,M = spimage.downsample(I,downsampling,mask=M)
    imsave("./%s/intensities_native.png" % index_number,numpy.log10(abs(img.image)+1)*numpy.log10(img.mask*10),vmin=1.5)
    imsave("./%s/intensities_downsampled.png" % index_number,numpy.log10(I+1)*numpy.log10(M*10),vmin=1.5)
    spimage.sp_image_free(img)
    return I,M

def reconstruct(I,M,S,N=None,index_number=None):
    R = spimage.Reconstructor()
    R.set_intensities(I,shifted=False)
    R.set_mask(M,shifted=False)
    R.set_number_of_iterations(4000)
    R.set_initial_support(radius=10)
    R.set_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=2000)
    R.append_phasing_algorithm("raar",beta_init=0.9,beta_final=0.9,constraints=["enforce_positivity"],number_of_iterations=1000)
    R.append_phasing_algorithm("er",constraints=["enforce_positivity"],number_of_iterations=1000)
    R.set_support_algorithm("area",blur_init=6.,blur_final=6.,area_init=float(S["area_init_1st"]),area_final=float(S["area_final_1st"]),update_period=100,number_of_iterations=2000,center_image=False)
    R.append_support_algorithm("area",blur_init=6.,blur_final=1.,area_init=float(S["area_init_2nd"]),area_final=float(S["area_final_2nd"]),update_period=100,number_of_iterations=1000,center_image=False)
    R.append_support_algorithm("static",number_of_iterations=1000,center_image=False)
    if N is None:
        R.set_number_of_outputs(100)
        O = R.reconstruct()
    else:
        R.set_number_of_outputs(3)
        O = R.reconstruct_loop(N)
        if index_number is not None:
            W = spimage.CXIWriter("./%s/rec.cxi" % index_number,N)
            W.write_stack(O)
            W.close()
    R.clear()
    return O

def plot_real_error(real_error_final,index_number,real_error_threshold=None):
    # Check real space errors
    real_error_final_s = numpy.copy(real_error_final[:])
    #rmax = 0.1
    #s = [s for s,r in zip(range(N_rec),real_error_final[:]) if r < rmax]
    real_error_final_s.sort()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(real_error_final_s)
    if real_error_threshold is not None:
        ax.axhline(real_error_threshold)
    ax.set_ylabel("Real space error")
    ax.set_xlabel("No. of reconstruction")
    #rs_max = abs(real_space_final[s,:,:]).max()
    fig.savefig("./%s/real_space_error.png" % index_number,dpi=300)

def plot_fourier_error(fourier_error_final,index_number,fourier_error_threshold=None):
    # Check fourier space errors
    fourier_error_final_s = numpy.copy(fourier_error_final[:])
    #rmax = 0.1
    #s = [s for s,r in zip(range(N_rec),fourier_error_final[:]) if r < rmax]
    fourier_error_final_s.sort()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(fourier_error_final_s)
    if fourier_error_threshold is not None:
        ax.axhline(fourier_error_threshold)
    ax.set_ylabel("Fourier space error")
    ax.set_xlabel("No. of reconstruction")
    #rs_max = abs(fourier_space_final[s,:,:]).max()
    fig.savefig("./%s/fourier_space_error.png" % index_number,dpi=300)

def imsave_reconstructions(real_space,fourier_error,index_number):
    s = fourier_error.argsort()
    vmax = abs(real_space).max()
    for i,rs,fe in zip(range(len(fourier_error)),real_space[s,:,:],fourier_error[s]):
        imsave("./%s/img_%04i_ferr_%.4f.png" % (index_number,i,fe),abs(rs),vmax=vmax,vmin=0.)

def prtf(real_space,support,index_number=None):
    prtf = spimage.prtf(real_space,support,
                        translate=True,enantio=True,full_out=True)
    if index_number is not None:
        with h5py.File("./%s/prtf.h5" % index_number,"w") as f:
            for k,v in prtf.items():
                f[k] = v
    return prtf

def plot_prtf(prtf,index_number,downsampling):
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
    fig = figure(figsize=(6,4))
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
    fig.savefig("./%s/prtf.png" % index_number,dpi=300)
    I = abs(prtf["super_image"])
    imsave("./%s/super_native.png" % index_number,I)
    from python_tools import imgtools,gentools,colormaptools
    Ic = imgtools.crop(I,40,"center_of_mass")
    imsave("./%s/super_cropped.png" % index_number,Ic)
    Icu = imgtools.upsample(Ic,20)
    imsave("./%s/super_cropped_upsampled.png" % index_number,Icu,cmap=colormaptools.cmaps["jet_lightbg2"])
