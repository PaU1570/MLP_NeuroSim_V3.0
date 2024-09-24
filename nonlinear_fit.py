import numpy as np
from scipy.optimize import curve_fit
import sys
import json
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# file is expected to be in the format:
# Setup title	pulsedAmplitudeSweep_DCIV
# test date	19.09.2024
# test time	14:20:07
# Device ID	LBE247_ID301XR1000Oct
# Count	1
# TriggerOutput	0
# TriggerPolarity	positive
# Measurement parameters
# groundWGFMU,pulseWGFMU,startVolage1,endVoltage1,startVolage2,endVoltage2,stepSize,riseTime,fallTime,pulseWidth,pulseDelay,VpreCond1,VpreCond2,riseTimePrecond,fallTimePrecon,pulseWidthPreCond,bothSides,repetitions,preConEveryLoop,groundVoltageDuringPulse,measName,
# SMU2,WGFMU1,0.000000,1.500000,0.000000,-1.500000,0.100000,0.000000020,0.000000020,0.005000000,0.001000000,0.000000,0.000000,0.000100000,0.000100000,0.005000000,1,2,0,0.000000,5e-3s-2V-m2V,
# Measurement Data
# index, pulse amplitude (V), R_low (ohm), R_high (ohm)
#     0,  0.000,6583278472.679,7662835249.042
#     1,  0.100,6587615283.267,7710100231.303
#     ...

def read_file(filename):
# read metadata and parameters from file
    with open(filename, 'r') as file:
        lines = file.readlines()
        # Extract metadata
        test_date = lines[1].split('\t')[1].strip()
        test_time = lines[2].split('\t')[1].strip()
        device_id = lines[3].split('\t')[1].strip()
        metadata = {'test_date': test_date, 'test_time': test_time, 'device_id': device_id}

        # Extract measurement parameters
        param_line = lines[8].strip().split(',')
        value_line = lines[9].strip().split(',')
        meas_params = dict()
        for param, value in zip(param_line, value_line):
            try:
                meas_params[param] = float(value)
            except ValueError:
                meas_params[param] = value

    # read measurement data
    meas_data = np.loadtxt(filename, delimiter=',', skiprows=12)

    return metadata, meas_params, meas_data

def plot_1(meas_data, kpos, kneg):
    """
    Plot the measured data with points to be deleted in red and the rest in blue.
    
    Args:
        meas_data: (np.array) Measured data
        kpos: (np.array) Indices of positive pulses
        kneg: (np.array) Indices of negative pulses
    """

    # not strictly necessary to use mask, as plotting the red points first will make them appear below the blue points,
    # but you can still see a slight red edge
    mask = np.ones(meas_data.shape[0], dtype=bool)
    mask[kpos] = False
    mask[kneg] = False

    fig, ax = plt.subplots()
    ax.plot(meas_data[mask,1], meas_data[mask,3], label='R_high', ls='none', marker='o', color='r')
    ax.plot(meas_data[kneg,1][0], meas_data[kneg,3][0], label='R_high', ls='none', marker='o', color='b')
    ax.plot(meas_data[kpos,1][0], meas_data[kpos,3][0], label='R_high', ls='none', marker='o', color='b')
    ax.set(xlabel='Pulse Amplitude (V)', ylabel='R_high (ohm)')
    plt.draw()

def nonlinear_fit(pulse_num_LTP, exp_LTP, pulse_num_LTD, exp_LTD, plotmode = 1, filename = None):
    """
    Fit and plot LTP and LTD data according to NeuroSim model.

    Args:
        pulse_num_LTP: (np.array) Normalized pulse number for LTP data
        exp_LTP: (np.array) Normalized experimental conductance data for LTP
        pulse_num_LTD: (np.array) Normalized pulse number for LTD data
        exp_LTD: (np.array) Normalized experimental conductance data for LTD

    Returns:
        best_A_LTP: (float) Best fit parameter A for LTP
        best_A_LTD: (float) Best fit parameter A for LTD
    """

    # fit LTP
    def model(pulse_number, A):
        B = 1./(1 - np.exp(-1./A))
        return B * (1 - np.exp(-pulse_number / A))
    
    popt_ltp, _ = curve_fit(model, pulse_num_LTP, exp_LTP, p0=[1.])
    popt_ltd, _ = curve_fit(model, pulse_num_LTD, exp_LTD, p0=[-1.])
    best_A_LTP = popt_ltp[0]
    best_A_LTD = popt_ltd[0]

    xdata = np.linspace(0, 1, 100)
    y_bestfit_LTP = model(xdata, best_A_LTP)
    y_bestfit_LTD = model(xdata, best_A_LTD)

    # compute r^2 values
    def r_squared(y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    
    r_squared_LTP = r_squared(exp_LTP, model(pulse_num_LTP, best_A_LTP))
    r_squared_LTD = r_squared(exp_LTD, model(pulse_num_LTD, best_A_LTD))

    if plotmode != 0:
        fig, ax = plt.subplots()
        ax.plot(pulse_num_LTP, exp_LTP, label='Exp. data (LTP)', ls='none', marker='o', color='b')
        ax.plot(pulse_num_LTD, exp_LTD, label='Exp. data (LTD)', ls='none', marker='o', color='r')
        ax.plot(xdata, y_bestfit_LTP, label=f'Best fit (LTP): A={best_A_LTP:.3f}, $R^2$={r_squared_LTP:.3f}', color='b')
        ax.plot(xdata, y_bestfit_LTD, label=f'Best fit (LTD): A={best_A_LTD:.3f}, $R^2$={r_squared_LTD:.3f}', color='r')
        ax.set(xlabel='Normalized Pulse Number', ylabel='Normalized Conductance')
        ax.legend()

        if plotmode == 1:
            plt.draw()
        else:
            plt.savefig(filename)

    return best_A_LTP, best_A_LTD

def map_A_to_NL(A):
    """
    Map the best fit parameter A to the corresponding nonlinearity label.

    Args:
        A: (float) Best fit parameter A (can be positive or negative)

    Returns:
        (float) Nonlinearity label (with the same sign as A)
    """
    # This normalized paramA table corresponds to nonlinearity label from 0.01 to 9, with step=0.01 (taken from formula.cpp)
    A_list = np.array([	126.268958,	63.134314,	42.089359,	31.566827,	25.253264,	21.044185,	
						18.037668,	15.782754,	14.028906,	12.625807,	11.477796,	10.521102,	
						9.711575,	9.017679,	8.416288,	7.890057,	7.425722,	7.012968,	
						6.643650,	6.311253,	6.010503,	5.737083,	5.487429,	5.258571,	
						5.048012,	4.853642,	4.673662,	4.506529,	4.350915,	4.205668,	
						4.069785,	3.942387,	3.822704,	3.710055,	3.603836,	3.503513,	
						3.408606,	3.318688,	3.233376,	3.152324,	3.075221,	3.001784,	
						2.931757,	2.864909,	2.801026,	2.739916,	2.681402,	2.625322,	
						2.571526,	2.519877,	2.470249,	2.422526,	2.376600,	2.332370,	
						2.289745,	2.248638,	2.208970,	2.170666,	2.133656,	2.097877,	
						2.063266,	2.029769,	1.997332,	1.965904,	1.935441,	1.905897,	
						1.877232,	1.849406,	1.822384,	1.796131,	1.770614,	1.745803,	
						1.721669,	1.698184,	1.675322,	1.653059,	1.631371,	1.610236,	
						1.589634,	1.569544,	1.549947,	1.530826,	1.512163,	1.493941,	
						1.476145,	1.458761,	1.441774,	1.425170,	1.408937,	1.393062,	
						1.377534,	1.362340,	1.347472,	1.332917,	1.318666,	1.304709,	
						1.291038,	1.277644,	1.264518,	1.251653,	1.239040,	1.226672,	
						1.214542,	1.202643,	1.190969,	1.179512,	1.168268,	1.157230,	
						1.146393,	1.135750,	1.125297,	1.115029,	1.104940,	1.095027,	
						1.085284,	1.075707,	1.066292,	1.057034,	1.047930,	1.038976,	
						1.030168,	1.021503,	1.012977,	1.004586,	0.996328,	0.988199,	
						0.980196,	0.972317,	0.964558,	0.956917,	0.949390,	0.941976,	
						0.934671,	0.927474,	0.920382,	0.913393,	0.906504,	0.899713,	
						0.893018,	0.886417,	0.879908,	0.873489,	0.867158,	0.860914,	
						0.854754,	0.848677,	0.842681,	0.836765,	0.830926,	0.825164,	
						0.819477,	0.813863,	0.808320,	0.802849,	0.797446,	0.792111,	
						0.786843,	0.781640,	0.776500,	0.771424,	0.766409,	0.761455,	
						0.756560,	0.751723,	0.746944,	0.742221,	0.737553,	0.732939,	
						0.728378,	0.723870,	0.719413,	0.715006,	0.710649,	0.706341,	
						0.702081,	0.697867,	0.693700,	0.689579,	0.685502,	0.681470,	
						0.677480,	0.673533,	0.669628,	0.665764,	0.661941,	0.658157,	
						0.654413,	0.650707,	0.647039,	0.643409,	0.639815,	0.636257,	
						0.632736,	0.629249,	0.625796,	0.622378,	0.618994,	0.615642,	
						0.612322,	0.609035,	0.605779,	0.602555,	0.599361,	0.596197,	
						0.593062,	0.589957,	0.586881,	0.583833,	0.580814,	0.577822,	
						0.574857,	0.571919,	0.569007,	0.566121,	0.563262,	0.560427,	
						0.557618,	0.554833,	0.552072,	0.549336,	0.546623,	0.543934,	
						0.541267,	0.538624,	0.536002,	0.533403,	0.530826,	0.528270,	
						0.525735,	0.523222,	0.520729,	0.518256,	0.515804,	0.513372,	
						0.510959,	0.508565,	0.506191,	0.503836,	0.501499,	0.499181,	
						0.496881,	0.494599,	0.492335,	0.490088,	0.487859,	0.485647,	
						0.483451,	0.481273,	0.479111,	0.476965,	0.474835,	0.472721,	
						0.470623,	0.468541,	0.466473,	0.464421,	0.462384,	0.460362,	
						0.458354,	0.456361,	0.454382,	0.452418,	0.450467,	0.448530,	
						0.446607,	0.444697,	0.442801,	0.440917,	0.439047,	0.437190,	
						0.435345,	0.433514,	0.431694,	0.429887,	0.428092,	0.426310,	
						0.424539,	0.422780,	0.421033,	0.419297,	0.417573,	0.415860,	
						0.414158,	0.412467,	0.410787,	0.409118,	0.407460,	0.405812,	
						0.404175,	0.402549,	0.400932,	0.399326,	0.397730,	0.396143,	
						0.394567,	0.393000,	0.391443,	0.389896,	0.388358,	0.386830,	
						0.385310,	0.383800,	0.382299,	0.380807,	0.379324,	0.377850,	
						0.376385,	0.374928,	0.373479,	0.372040,	0.370608,	0.369185,	
						0.367770,	0.366363,	0.364965,	0.363574,	0.362192,	0.360817,	
						0.359450,	0.358090,	0.356738,	0.355394,	0.354057,	0.352728,	
						0.351406,	0.350091,	0.348784,	0.347484,	0.346190,	0.344904,	
						0.343625,	0.342352,	0.341087,	0.339828,	0.338576,	0.337330,	
						0.336091,	0.334859,	0.333632,	0.332413,	0.331200,	0.329993,	
						0.328792,	0.327597,	0.326409,	0.325226,	0.324050,	0.322879,	
						0.321715,	0.320556,	0.319403,	0.318256,	0.317114,	0.315979,	
						0.314848,	0.313724,	0.312605,	0.311491,	0.310382,	0.309280,	
						0.308182,	0.307090,	0.306003,	0.304921,	0.303844,	0.302772,	
						0.301705,	0.300644,	0.299587,	0.298536,	0.297489,	0.296447,	
						0.295410,	0.294377,	0.293350,	0.292327,	0.291308,	0.290295,	
						0.289285,	0.288281,	0.287281,	0.286285,	0.285294,	0.284307,	
						0.283325,	0.282347,	0.281373,	0.280403,	0.279438,	0.278477,	
						0.277520,	0.276567,	0.275618,	0.274673,	0.273733,	0.272796,	
						0.271863,	0.270935,	0.270010,	0.269089,	0.268171,	0.267258,	
						0.266349,	0.265443,	0.264541,	0.263642,	0.262747,	0.261856,	
						0.260969,	0.260085,	0.259205,	0.258328,	0.257454,	0.256585,	
						0.255718,	0.254855,	0.253996,	0.253140,	0.252287,	0.251437,	
						0.250591,	0.249748,	0.248908,	0.248072,	0.247239,	0.246409,	
						0.245582,	0.244758,	0.243937,	0.243120,	0.242305,	0.241494,	
						0.240685,	0.239880,	0.239077,	0.238278,	0.237481,	0.236687,	
						0.235897,	0.235109,	0.234324,	0.233541,	0.232762,	0.231985,	
						0.231212,	0.230440,	0.229672,	0.228906,	0.228143,	0.227383,	
						0.226626,	0.225871,	0.225118,	0.224368,	0.223621,	0.222877,	
						0.222134,	0.221395,	0.220658,	0.219923,	0.219191,	0.218462,	
						0.217734,	0.217010,	0.216287,	0.215568,	0.214850,	0.214135,	
						0.213422,	0.212711,	0.212003,	0.211297,	0.210594,	0.209892,	
						0.209193,	0.208496,	0.207802,	0.207109,	0.206419,	0.205731,	
						0.205045,	0.204361,	0.203680,	0.203000,	0.202323,	0.201648,	
						0.200975,	0.200303,	0.199634,	0.198967,	0.198302,	0.197639,	
						0.196978,	0.196319,	0.195662,	0.195007,	0.194354,	0.193703,	
						0.193054,	0.192406,	0.191761,	0.191117,	0.190476,	0.189836,	
						0.189198,	0.188562,	0.187928,	0.187295,	0.186664,	0.186036,	
						0.185409,	0.184783,	0.184160,	0.183538,	0.182918,	0.182300,	
						0.181683,	0.181068,	0.180455,	0.179843,	0.179234,	0.178625,	
						0.178019,	0.177414,	0.176811,	0.176209,	0.175609,	0.175011,	
						0.174414,	0.173819,	0.173226,	0.172634,	0.172043,	0.171454,	
						0.170867,	0.170281,	0.169697,	0.169114,	0.168533,	0.167953,	
						0.167375,	0.166798,	0.166222,	0.165649,	0.165076,	0.164505,	
						0.163936,	0.163368,	0.162801,	0.162236,	0.161672,	0.161109,	
						0.160548,	0.159989,	0.159430,	0.158873,	0.158318,	0.157764,	
						0.157211,	0.156659,	0.156109,	0.155560,	0.155013,	0.154466,	
						0.153921,	0.153378,	0.152835,	0.152294,	0.151755,	0.151216,	
						0.150679,	0.150143,	0.149608,	0.149075,	0.148542,	0.148011,	
						0.147481,	0.146953,	0.146425,	0.145899,	0.145374,	0.144850,	
						0.144328,	0.143806,	0.143286,	0.142767,	0.142249,	0.141732,	
						0.141217,	0.140702,	0.140189,	0.139676,	0.139165,	0.138655,	
						0.138147,	0.137639,	0.137132,	0.136627,	0.136122,	0.135619,	
						0.135117,	0.134616,	0.134116,	0.133617,	0.133119,	0.132622,	
						0.132126,	0.131631,	0.131138,	0.130645,	0.130153,	0.129663,	
						0.129173,	0.128685,	0.128197,	0.127711,	0.127225,	0.126741,	
						0.126258,	0.125775,	0.125294,	0.124813,	0.124334,	0.123855,	
						0.123378,	0.122901,	0.122426,	0.121951,	0.121478,	0.121005,	
						0.120533,	0.120063,	0.119593,	0.119124,	0.118656,	0.118189,	
						0.117723,	0.117258,	0.116794,	0.116331,	0.115869,	0.115407,	
						0.114947,	0.114487,	0.114029,	0.113571,	0.113114,	0.112659,	
						0.112204,	0.111750,	0.111296,	0.110844,	0.110393,	0.109942,	
						0.109493,	0.109044,	0.108596,	0.108149,	0.107703,	0.107258,	
						0.106813,	0.106370,	0.105927,	0.105486,	0.105045,	0.104605,	
						0.104166,	0.103727,	0.103290,	0.102853,	0.102417,	0.101982,	
						0.101548,	0.101115,	0.100683,	0.100251,	0.099820,	0.099390,	
						0.098961,	0.098533,	0.098105,	0.097679,	0.097253,	0.096828,	
						0.096404,	0.095981,	0.095558,	0.095136,	0.094715,	0.094295,	
						0.093876,	0.093458,	0.093040,	0.092623,	0.092207,	0.091792,	
						0.091377,	0.090964,	0.090551,	0.090139,	0.089728,	0.089317,	
						0.088907,	0.088498,	0.088090,	0.087683,	0.087276,	0.086871,	
						0.086466,	0.086062,	0.085658,	0.085256,	0.084854,	0.084453,	
						0.084052,	0.083653,	0.083254,	0.082856,	0.082459,	0.082062,	
						0.081667,	0.081272,	0.080878,	0.080484,	0.080092,	0.079700,	
						0.079309,	0.078919,	0.078529,	0.078140,	0.077752,	0.077365,	
						0.076979,	0.076593,	0.076208,	0.075824,	0.075440,	0.075057,	
						0.074675,	0.074294,	0.073914,	0.073534,	0.073155,	0.072777,	
						0.072400,	0.072023,	0.071647,	0.071272,	0.070897,	0.070524,	
						0.070151,	0.069778,	0.069407,	0.069036,	0.068666,	0.068297,	
						0.067929,	0.067561,	0.067194,	0.066827,	0.066462,	0.066097,	
						0.065733,	0.065370,	0.065007,	0.064645,	0.064284,	0.063924,	
						0.063564,	0.063206,	0.062847,	0.062490,	0.062133,	0.061777,	
						0.061422,	0.061068,	0.060714,	0.060361,	0.060009,	0.059657,	
						0.059306,	0.058956,	0.058607,	0.058259,	0.057911,	0.057564,	
						0.057217,	0.056871,	0.056527,	0.056182,	0.055839,	0.055496,	
						0.055154,	0.054813,	0.054472,	0.054132,	0.053793,	0.053455,	
						0.053117,	0.052781,	0.052444,	0.052109,	0.051774,	0.051440,	
						0.051107,	0.050774,	0.050443,	0.050112,	0.049781,	0.049452,	
						0.049123,	0.048795,	0.048467,	0.048141,	0.047815,	0.047489,	
						0.047165,	0.046841,	0.046518,	0.046196,	0.045874,	0.045553,	
						0.045233,	0.044914,	0.044595,	0.044277,	0.043960,	0.043643,	
						0.043328,	0.043013,	0.042698,	0.042385,	0.042072,	0.041760,	
						0.041449,	0.041138,	0.040828,	0.040519,	0.040211,	0.039903,	
						0.039596,	0.039290,	0.038984,	0.038680,	0.038376,	0.038072,	
						0.037770,	0.037468,	0.037167,	0.036867,	0.036567,	0.036268,	
						0.035970,	0.035673,	0.035376,	0.035080,	0.034785,	0.034491,	
						0.034197,	0.033904,	0.033612,	0.033321,	0.033030,	0.032740,	
						0.032451,	0.032163,	0.031875,	0.031588,	0.031302,	0.031016,	
						0.030732,	0.030448,	0.030165,	0.029882,	0.029601,	0.029320,	
						0.029040,	0.028760,	0.028482,	0.028204,	0.027927,	0.027651,	
						0.027375,	0.027101,	0.026827,	0.026553,	0.026281,	0.026009,	
						0.025738,	0.025468,	0.025199,	0.024930,	0.024663,	0.024396,	
						0.024129,	0.023864,	0.023599,	0.023336,	0.023073,	0.022810])
    sign = 1 if A > 0 else -1
    # find the index of the closest value in the list
    diff = np.abs(A_list - sign*A)
    min_index = np.argmin(diff)
    # convert to nonlinearity label
    return sign * 0.01 * (min_index + 1)
    

def generate_config(meas_data, kpos, kneg, params, device_type='RealDevice', filename=None):
    """
    Generate a JSON configuration file based on the measured data and the best fit parameters.

    Args:
        meas_data: (np.array) Measured data
        kpos: (np.array) Indices of positive pulses
        kneg: (np.array) Indices of negative pulses
        params: (dict) Measurement parameters: readVoltage, twidth, VStartNeg, VEndNeg, VStartPos, VEndPos, stepSizeLTD, stepSizeLTP, NL, best_NL_LTP, best_NL_LTD, sigmaCtoC, sigmaDtoD
        device_type: (str) Type of device (currently only 'RealDevice' is supported) TODO: add more device types
    """

    # load default config
    script_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_path, 'config-defaults.json')
    with open(config_path, 'r') as file:
        config = json.load(file)
  
    # delete configuration for other device types to keep it uncluttered
    config['device-params']['type'] = device_type
    keys_to_delete = []
    for key in config['device-params']:
        if key != 'type' and key != device_type:
            keys_to_delete.append(key)
    for key in keys_to_delete:
        del config['device-params'][key]

    maxConductance = 1./min(meas_data[kpos,3][0])
    minConductance = 1./max(meas_data[kpos,3][0])

    config['device-params'][device_type]['maxConductance'] = maxConductance
    config['device-params'][device_type]['minConductance'] = minConductance
    config['device-params'][device_type]['avgMaxConductance'] = maxConductance
    config['device-params'][device_type]['avgMinConductance'] = minConductance
    config['device-params'][device_type]['readVoltage'] = params['readVoltage']
    config['device-params'][device_type]['readPulseWidth'] = params['twidth']
    config['device-params'][device_type]['writeVoltageLTP'] = params['VEndNeg']
    config['device-params'][device_type]['writeVoltageLTD'] = params['VEndPos']
    config['device-params'][device_type]['writePulseWidthLTP'] = params['twidth']
    config['device-params'][device_type]['writePulseWidthLTD'] = params['twidth']
    config['device-params'][device_type]['maxNumLevelLTP'] = len(kneg[0])
    config['device-params'][device_type]['maxNumLevelLTD'] = len(kpos[0])
    config['device-params'][device_type]['NL'] = params['NL'] # this is not used if nonlinearIV = False
    config['device-params'][device_type]['nonIdenticalPulse']['enabled'] = True
    config['device-params'][device_type]['nonIdenticalPulse']['VinitLTP'] = params['VStartNeg']
    config['device-params'][device_type]['nonIdenticalPulse']['VstepLTP'] = params['stepSizeLTP']
    config['device-params'][device_type]['nonIdenticalPulse']['VinitLTD'] = params['VStartPos']
    config['device-params'][device_type]['nonIdenticalPulse']['VstepLTD'] = params['stepSizeLTD']
    config['device-params'][device_type]['nonIdenticalPulse']['PWinitLTP'] = params['twidth']
    config['device-params'][device_type]['nonIdenticalPulse']['PWinitLTD'] = params['twidth']
    config['device-params'][device_type]['nonIdenticalPulse']['PWstepLTP'] = 0
    config['device-params'][device_type]['nonIdenticalPulse']['PWstepLTD'] = 0
    config['device-params'][device_type]['weightUpdateVariationParams']['NL_LTP'] = params['best_NL_LTP']
    config['device-params'][device_type]['weightUpdateVariationParams']['NL_LTD'] = params['best_NL_LTD']
    config['device-params'][device_type]['weightUpdateVariationParams']['sigmaDtoD'] = params['sigmaDtoD']
    config['device-params'][device_type]['weightUpdateVariationParams']['sigmaCtoC'] = params['sigmaCtoC']

    if filename is None:
        print(json.dumps(config, indent=4))
    else:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=4)
        print("Configuration file written to", filename)


if __name__ == '__main__':
    # read filename from command line
    if len(sys.argv) < 2:
        print("Usage: python nonlinear_fit.py <filename> [noplot | saveplot (optional)]")
        sys.exit(1)

    filename = sys.argv[1]
    metadata, meas_params, meas_data = read_file(filename)

    results_folder = os.path.join(os.path.dirname(filename), 'Results')
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    plotmode = 1 # 0: no plot, 1: plot, 2: save plot
    if len(sys.argv) == 3 and sys.argv[2] == 'noplot':
        plotmode = 0
    if len(sys.argv) == 3 and sys.argv[2] == 'saveplot':
        plotmode = 2

    for key, value in metadata.items():
        print(f"{key}: {value}")

    for key, value in meas_params.items():
        print(f"{key}: {value}")

    VStartPos = max(0.1, meas_params['startVolage1']) # (sic)
    VEndPos = min(4.1, meas_params['endVoltage1'])
    VStartNeg = min(0, meas_params['startVolage2']) # (sic)
    VEndNeg = max(-1.4, meas_params['endVoltage2'])
    stepSize = meas_params['stepSize']
    twidth = meas_params['pulseWidth']

    kpos = np.where((meas_data[:,1] > VStartPos) & (meas_data[:,1] < VEndPos))
    kneg = np.where((meas_data[:,1] < VStartNeg) & (meas_data[:,1] > VEndNeg))

    if plotmode == 1:
        plot_1(meas_data, kpos, kneg)

    # get on/off ratio
    onOffRatio = max(meas_data[kpos,3][0]) / min(meas_data[kpos,3][0])

    # write results to file
    allResults_filename = os.path.join(results_folder, os.path.basename(filename.replace('.csv', '_AllResults.dat')))
    with open(allResults_filename, 'w') as f:
        f.write(f"VStartPos={VStartPos} V, VEndPos={VEndPos} V, VStartNeg={VStartNeg} V, VEndNeg={VEndNeg} V, twidth={twidth} s, onOffRatio={onOffRatio}\n")
        f.write("Pulse Number,index,Pulse Amplitude (V),R_low (ohm),R_high (ohm)\n")
        for i, (index, pulse_amplitude, R_low, R_high) in enumerate(np.vstack((meas_data[kpos,:][0], meas_data[kneg,:][0]))):
            f.write(f"{i+1},{int(index)},{pulse_amplitude},{R_low},{R_high}\n")


    exp_LTD_raw = np.flip(1. / meas_data[kpos,3][0])
    exp_LTP_raw = 1. / meas_data[kneg,3][0]
    pulse_num_LTD_raw = np.linspace(0, len(exp_LTD_raw) - 1, len(exp_LTD_raw))
    pulse_num_LTP_raw = np.linspace(0, len(exp_LTP_raw) - 1, len(exp_LTP_raw))
    # normalize
    exp_LTD_norm = (exp_LTD_raw - min(exp_LTD_raw))/(max(exp_LTD_raw) - min(exp_LTD_raw))
    exp_LTP_norm = (exp_LTP_raw - min(exp_LTP_raw))/(max(exp_LTP_raw) - min(exp_LTP_raw))
    pulse_num_LTD_norm = pulse_num_LTD_raw / max(pulse_num_LTD_raw)
    pulse_num_LTP_norm = pulse_num_LTP_raw / max(pulse_num_LTP_raw)

    plot_filename = os.path.join(results_folder, os.path.basename(filename.replace('.csv', '_fit.png')))
    best_A_LTP, best_A_LTD = nonlinear_fit(pulse_num_LTP_norm, exp_LTP_norm, pulse_num_LTD_norm, exp_LTD_norm, plotmode=plotmode, filename=plot_filename)
    print(f"Best fit parameter A for LTP: {best_A_LTP:.3f}")
    print(f"Best fit parameter A for LTD: {best_A_LTD:.3f}")

    best_NL_LTP = map_A_to_NL(best_A_LTP)
    best_NL_LTD = map_A_to_NL(best_A_LTD)
    print(f"Best fit nonlinearity label for LTP: {best_NL_LTP:.2f}")
    print(f"Best fit nonlinearity label for LTD: {best_NL_LTD:.2f}")

    params = {
        'readVoltage': 0.1,
        'twidth': twidth,
        'VStartNeg': VStartNeg,
        'VEndNeg': VEndNeg,
        'VStartPos': VStartPos,
        'VEndPos': VEndPos,
        'stepSizeLTP': -stepSize,
        'stepSizeLTD': stepSize,
        'NL': 40,
        'best_NL_LTP': best_NL_LTP,
        'best_NL_LTD': best_NL_LTD,
        'sigmaCtoC': 0.05,
        'sigmaDtoD': 0.05
    }

    config_filename = os.path.join(results_folder, os.path.basename(filename.replace('.csv', '.json')))
    generate_config(meas_data, kpos, kneg, params, filename=config_filename)

    if plotmode == 1:
        print("Waiting for plots to close to terminate program...")
        plt.show()