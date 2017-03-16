# General imports
import os,sys
import time, numpy
import datetime
import scipy.constants as constants
import logging
import h5py
import csv

# Hummingbird imports
import analysis.event
import analysis.pixel_detector
import analysis.beamline
import analysis.hitfinding
import utils.reader
import utils.cxiwriter
import utils.array
import ipc.mpi
from backend import add_record

# Local imports
this_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(this_dir)
from quad_correction import fix_upper_right_quadrant

# Set logging level of h5writer
utils.cxiwriter.h5writer.logger.setLevel("INFO")

# Additional commandline arguments
from utils.cmdline_args import argparser, add_config_file_argument
add_config_file_argument('--out-dir', metavar='out_dir', nargs='?',
                         help="Output directory", required=True,
                         type=str)
add_config_file_argument('--hitscore-threshold', metavar='hitscore_threshold',
                         help="Hitscore threshold [if not provided read from CSV file]",
                         type=int)
add_config_file_argument('--photon-energy-ev', metavar='photon_energy_ev',
                         help="Manually set nominal photon energy in unit electron volts (used for example for pnCCD gain calculation and hit finding) [if not provided read from CSV file]",
                         type=float)
add_config_file_argument('--rescaling-factors-asics', metavar='rescaling_factors_asics',
                         help="Manually set the 4 rescaling factors for upper right quadrant (for example 2.0,1.0,1.0,1.0) [if not provided read from CSV file]",
                         type=str)
add_config_file_argument('--output-level',
                         help='Output level defines how much data per event will be stored (default=3, min=0, max=3)', 
                         type=int, default=3)
add_config_file_argument('--do-raw',
                         help="Only do pedestal correction, no other corrections",
                         type=int, default=0)
add_config_file_argument('--do-quad',
                         help="Correct artefacts in \"upper right\" quadrant",
                         type=int, default=1)
add_config_file_argument('--do-cmc',
                         help="Apply common mode correction along \"horizontal\" lines",
                         type=int, default=1)
add_config_file_argument('--do-metrology',
                         help="Move pixels to their physical locations",
                         type=int, default=1)
add_config_file_argument('--check-sha-amol3416',
                         help="Abort processing if provided SHA key does not match SHA key of loaded amol3416 repo.",
                         type=str, default=None)
add_config_file_argument('--check-sha-hummingbird',
                         help="Abort processing if provided SHA key does not match SHA key of loaded hummingbird repo.",
                         type=str, default=None)
add_config_file_argument('--do-metrology',
                         help="Move pixels to their physical locations",
                         type=int, default=1)
args = argparser.parse_args()

# The way how the output level argument is interpreted
save_anything = args.output_level > 0
save_tof = args.output_level >= 2
save_pnccd = args.output_level >= 3

# Skipping events with no FEL pulse
do_skipdark = True

# ------------------------------
# PSANA
# ------------------------------
state = {}
state['LCLS/DataSource'] = 'exp=amol3416:run=%d' %args.lcls_run_number
state['LCLS/PsanaConf'] = this_dir + '/pnccd_front.cfg'

# PNCCD Detector
front_type = "calibrated"
front_key  = "pnccdFront[%s]" % front_type

# Gain
gain = None
gain_mode = None
gain_mode_key = "PNCCD:FRONT:GAIN" 

# INJECTOR MOTORS
injector_x_key = "AMO:PPL:MMS:01.RBV"
injector_z_key = "AMO:PPL:MMS:03.RBV"

# NOZZLE PRESSURES
nozzle_pressure_1_key = "AMO:SDS:PAREG:02:PRESS"
nozzle_pressure_2_key = "AMO:SDS:PAREG:03:PRESS"

# INJECTOR PRESSURES
injector_pressure_1_key = "AMO:LMP:VG:43:PRESS"
injector_pressure_2_key = "AMO:LMP:VG:40:PRESS"
injector_pressure_3_key = ""

# INJECTOR VOLTAGE
injector_voltage_key = "IOC:AMO:HV1:VHS2:CH0:VoltageMeasure"
injector_current_key = "IOC:AMO:HV1:VHS2:CH0:CurrentMeasure"

# ELECTROSPRAY
electrospray_key = "AMO:PPL:MMS:18.RBV"

# PULSE LENGTH
pulselength_key = "SIOC:SYS0:ML00:AO820"

# PNCCD STAGE
pnccd_motorx = "AMO:LMP:MMS:09.RBV"
pnccd_motory_top = "AMO:LMP:MMS:07.RBV"
pnccd_motory_bottom = "AMO:LMP:MMS:08.RBV"
pnccd_motorz = "AMO:LMP:MMS:10.RBV"

# TOF DETECTOR
acqiris_type = "ionTOFs"
acqiris_key  = "Acqiris 2 Channel 0"
        
# For CXI writer we reduce the number of event readers by one
state['reduce_nr_event_readers'] = 1

# Run-specific parameters (these default values are for run 98 and 100)
hitscore_threshold = 3500
nominal_photon_energy_eV = 800
rescaling_factors_asics = [2,2,1,1]
dx_front = 73
dy_front = -3
dy_timing_gap = 0
run_type = "Sample"

# Overwrite parameter from the command line
if args.hitscore_threshold is not None:
    hitscore_threshold = args.hitscore_threshold
if args.photon_energy_ev is not None:
    nominal_photon_energy_eV = args.photon_energy_ev
if args.rescaling_factors_asics is not None:
    rescaling_factors_asics = numpy.array([numpy.float64(f) for f in args.rescaling_factors_asics.split(',')], dtype='float64')

# Mask
mask_front_templ = utils.reader.MaskReader(this_dir + "/mask_front.h5","/data/data").boolean_mask
(ny_front,nx_front) = mask_front_templ.shape
caught_mask_flag = False

# Counters
event_number = 0
hitcounter = 0

# Output filename
W = None
out_dir = args.out_dir
tmpfile  = '%s/.%s_r%04d_ol%i.h5' % (out_dir, data_mode, conf.run_nr, args.output_level)
donefile = '%s/%s_r%04d_ol%i.h5' % (out_dir, data_mode, conf.run_nr, args.output_level)
D_solo = {}

# Set up file writer
def beginning_of_run():
    global W
    W = utils.cxiwriter.CXIWriter(tmpfile, chunksize=10, compression=None)
    
# ---------------------------------------------------------
# E V E N T   C A L L
# ---------------------------------------------------------
def onEvent(evt):
    global event_number
    global hitcounter
    global gain
    global gain_mode
    global caught_mask_flag

    ft = front_type
    fk = front_key  
    event_number += 1

    # Time measurement
    analysis.event.printProcessingRate()

    # Average pulse energies, in case there is no pulse energy it returns None
    analysis.beamline.averagePulseEnergy(evt, evt["pulseEnergies"]) 
    # Skip frames that do not have a pulse energy
    try:
        pulse_energy = evt['analysis']['averagePulseEnergy']
        pulse = True
    except (TypeError,KeyError):
        pulse = False
    if do_skipdark and not pulse:
        print "No pulse energy. Skipping event."
        return

    # Average pulse energies, in case there is no pulse energy it returns None 
    analysis.beamline.averagePhotonEnergy(evt, evt["photonEnergies"])
    # Skip frames that do not have a photon energy
    try:
        evt['analysis']['averagePhotonEnergy']
    except (TypeError,KeyError):
        print "No photon energy. Skipping event."
        return       

    # Set to nominal photon energy
    photon_energy_ev_avg = evt['analysis']['averagePhotonEnergy']
    photon_energy_ev_nom = add_record(evt["analysis"], "analysis", "nominalPhotonEnergy" , nominal_photon_energy_eV)
    
    # Gain of PNCCD
    try:
        gain_mode = evt['parameters'][gain_mode_key].data
    except:
        print "Could not read gain mode from data stream. Skipping event."
        return
    analysis.pixel_detector.pnccdGain(evt, photon_energy_ev_nom, gain_mode)
    gain = evt['analysis']['gain'].data

    # Skip frames that do not have the pnCCDs
    try:
        evt[ft][fk]
    except (TypeError,KeyError):
        print "No front pnCCD. Skipping event."
        return

    front = evt[ft][fk]
    mask_front = mask_front_templ.copy()

    # We only need to copy the data if we have to manipulate it
    front_is_read_only = True

    # Data sometimes comes in 3D shape, making sure that preprocessing is only on 2D data of identical shape
    if len(front.data.shape) == 3:
        tmp = numpy.zeros(shape=(1024, 1024), dtype=front.data.dtype)
        # Left
        tmp[:512,:512] = front.data[0,:,:]
        tmp[512:,:512] = front.data[1,::-1,::-1]
        # Right
        tmp[:512,512:] = front.data[3,:,:]
        tmp[512:,512:] = front.data[2,::-1,::-1]
        front_is_read_only = False
    else:
        tmp = front.data
        
    ft = "analysis"
    fk = "preprocessed - " + fk
    add_record(evt[ft], ft, fk , tmp)
    front = evt[ft][fk]

    if not args.do_raw:        
        if args.do_quad:
            if front_is_read_only:
                front.data = front.data.copy()
                front_is_read_only = False
            # Compensate for artefacts in "upper right" quadrant
            fix_upper_right_quadrant(img_quad=front.data[:512, 512:], msk_quad=mask_front[:512, 512:], rescaling_factors_asics=rescaling_factors_asics)

        # Common mode correction for PNCCD
        if args.do_cmc and not args.do_raw:
            analysis.pixel_detector.commonModePNCCD2(evt, ft, fk, signal_threshold=100, min_nr_pixels_per_median=100)
            ft = "analysis"
            fk  = "corrected - " + fk
            front = evt[ft][fk]
            
        if args.do_metrology:
            if front_is_read_only:
                front.data = front.data.copy()
                front_is_read_only = False
            
            # Fix wrong wiring: Move "upper" quadrants by one pixel inwards
            # "Top left"
            front.data[:512,1:512] = front.data[:512,:512-1]
            mask_front[:512,0] = False
            # "Top right"
            front.data[:512,512:-1] = front.data[:512,512+1:]
            mask_front[:512,-1] = False

            if dy_timing_gap > 0:
                # Add vertical gap (timing problems in certain runs)
                front.data[:512,:] = front.data[:512,:]
                front.data[512+dy_timing_gap:,:] = front.data[512:-dy_timing_gap,:]
                front.data[512:512+dy_timing_gap,:] = 0
                mask_front[:512,:] = mask_front[:512,:]
                mask_front[512+dy_timing_gap:,:] = mask_front[512:-dy_timing_gap,:]
                mask_front[512:512+dy_timing_gap,:] = False
                
            # Move right detector half (assemble)
            front = analysis.pixel_detector.moveHalf(evt, front, horizontal=dx_front, vertical=dy_front, outkey='data_half-moved')
            ft = "analysis"
            fk  = "data_half-moved"
            mask_front = analysis.pixel_detector.moveHalf(evt, add_record(evt["analysis"], "analysis", "mask",mask_front), horizontal=dx_front, vertical=dy_front, outkey='mask_half-moved').data
        
    # Finding hits
    analysis.hitfinding.countLitPixels(evt, front, aduThreshold=gain, 
                                       hitscoreThreshold=hitscore_threshold, mask=mask_front)
    hit = evt["analysis"]["litpixel: isHit"].data
    if hit: hitcounter += 1

    # Saving to file
    if hit and save_anything:
        D = {}

        D["entry_1"] = {}

        if save_pnccd:
            D["entry_1"]["detector_1"] = {}
            D["entry_1"]["detector_2"] = {}

        D["entry_1"]["detector_3"] = {}
        D["entry_1"]["event"] = {}
        D["entry_1"]["injector"] = {}
        D["entry_1"]["FEL"] = {}
        D["entry_1"]["result_1"] = {}


        if save_pnccd:
            # PNCCD
            D["entry_1"]["detector_1"]["data"] = numpy.asarray(evt[ft][fk].data, dtype='float16')
            if ipc.mpi.is_main_event_reader() and len(D_solo) == 0:
                bitmask = numpy.array(mask_front, dtype='uint16')
                bitmask[bitmask==0] = 512
                bitmask[bitmask==1] = 0
                D_solo["entry_1"] = {}
                D_solo["entry_1"]["detector_1"] = {}
                D_solo["entry_1"]["detector_1"]["mask"] = bitmask
                caught_mask_flag = True
            D["entry_1"]["detector_1"]["gain"] = gain
            D["entry_1"]["detector_1"]["gain_mode"] = gain_mode
            D["entry_1"]["detector_1"]["rescaling_factors_asics"] = rescaling_factors_asics

        if save_tof:
            # TOF
            D["entry_1"]["detector_2"]["TOF"] = evt[acqiris_type][acqiris_key].data
        
        # GMD
        D["entry_1"]["detector_3"]["pulse_energy_mJ"] = pulse_energy.data
        
        # HIT PARAMETERS
        D["entry_1"]["result_1"]["hitscore_litpixel"] = evt["analysis"]["litpixel: hitscore"].data
        D["entry_1"]["result_1"]["hitscore_litpixel_threshold"] = hitscore_threshold

        # EVENT IDENTIFIERS
        D["entry_1"]["event"]["timestamp"] = evt["eventID"]["Timestamp"].timestamp
        D["entry_1"]["event"]["timestamp2"] =  evt["eventID"]["Timestamp"].timestamp2
        D["entry_1"]["event"]["fiducial"] = evt["eventID"]["Timestamp"].fiducials

        # INJECTOR
        D["entry_1"]["injector"]["injector_position_x"] = evt["parameters"][injector_x_key].data
        D["entry_1"]["injector"]["injector_position_y"] = evt["parameters"][injector_z_key].data
        D["entry_1"]["injector"]["injector_pressures_1"] = evt["parameters"][injector_pressure_1_key].data
        D["entry_1"]["injector"]["injector_pressures_2"] = evt["parameters"][injector_pressure_2_key].data
        D["entry_1"]["injector"]["nozzle_pressures_1"] = evt["parameters"][nozzle_pressure_1_key].data
        D["entry_1"]["injector"]["nozzle_pressures_2"] = evt["parameters"][nozzle_pressure_2_key].data
        D["entry_1"]["injector"]["injector_voltage"] = evt["parameters"][injector_voltage_key].data
        D["entry_1"]["injector"]["injector_current"] = evt["parameters"][injector_current_key].data
        D["entry_1"]["injector"]["electrospray_key"] = evt["parameters"][electrospray_key].data
    
        # FEL
        D["entry_1"]["FEL"]["photon_energy_eV_nominal"] = photon_energy_ev_nom.data
        D["entry_1"]["FEL"]["wavelength_nm_nominal"] = constants.c*constants.h/(photon_energy_ev_nom.data*constants.e)/1E-9
        D["entry_1"]["FEL"]["photon_energy_eV_SLAC"] = photon_energy_ev_avg.data
        D["entry_1"]["FEL"]["wavelength_nm_SLAC"] = constants.c*constants.h/(photon_energy_ev_avg.data*constants.e)/1E-9
        D["entry_1"]["FEL"]["pulse_length"] = evt["parameters"][pulselength_key].data

        W.write_slice(D)

def end_of_run():
    if ipc.mpi.is_main_event_reader():
        if "entry_1" not in D_solo:
            D_solo["entry_1"] = {}
        D_solo["entry_1"]["run_type"] = run_type
        W.write_solo(D_solo)
        if save_pnccd and not caught_mask_flag:
            print "WARNING: No mask was saved. You might want to reduce the number of processes or increase the number of frames to be processed."
    W.close(barrier=True)

    if ipc.mpi.is_main_event_reader():
        if save_anything:
            with h5py.File(tmpfile, "r+") as f:
                f["/entry_1/detector_3/data"] = h5py.SoftLink('/entry_1/detector_3/pulse_energy_mJ')
                if save_pnccd:
                    f["/entry_1/data_1"] = h5py.SoftLink('/entry_1/detector_1')
                if save_tof:
                    f["/entry_1/data_2"] = h5py.SoftLink('/entry_1/detector_2')
                    f["/entry_1/detector_2/data"] = h5py.SoftLink('/entry_1/detector_2/TOF')
                print "Successfully wrote soft links."
        os.system('mv %s %s' %(tmpfile, donefile))
        print "Moved temporary file %s to %s." % (tmpfile, donefile)
        print "Clean exit."
