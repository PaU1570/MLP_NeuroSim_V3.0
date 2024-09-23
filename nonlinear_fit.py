import numpy as np
from scipy.optimize import curve_fit
import sys
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
    # not strictly necessary to use mask, as plotting the red points first will make them appear below the blue points, but you can still see a slight red edge
    mask = np.ones(meas_data.shape[0], dtype=bool)
    mask[kpos] = False
    mask[kneg] = False
    fig, ax = plt.subplots()
    ax.plot(meas_data[mask,1], meas_data[mask,3], label='R_high', ls='none', marker='o', color='r')
    ax.plot(meas_data[kneg,1][0], meas_data[kneg,3][0], label='R_high', ls='none', marker='o', color='b')
    ax.plot(meas_data[kpos,1][0], meas_data[kpos,3][0], label='R_high', ls='none', marker='o', color='b')
    ax.set(xlabel='Pulse Amplitude (V)', ylabel='R_high (ohm)')
    plt.draw()

def nonlinear_fit(pulse_num_LTP, exp_LTP, pulse_num_LTD, exp_LTD):
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
    def ltp_model(pulse_number, A_LTP):
        B_LTP = 1./(1 - np.exp(-1./A_LTP))
        return B_LTP * (1 - np.exp(-pulse_number / A_LTP))
    
    def ltd_model(pulse_number, A_LTD):
        B_LTD = 1./(1 - np.exp(-1./A_LTD))
        return -B_LTD * (1 - np.exp((pulse_number - 1) / A_LTD)) + 1
    
    popt_ltp, _ = curve_fit(ltp_model, pulse_num_LTP, exp_LTP, p0=[1.])
    popt_ltd, _ = curve_fit(ltd_model, pulse_num_LTD, exp_LTD, p0=[1.])
    best_A_LTP = popt_ltp[0]
    best_A_LTD = popt_ltd[0]

    xdata = np.linspace(0, 1, 100)
    y_bestfit_LTP = ltp_model(xdata, best_A_LTP)
    y_bestfit_LTD = ltd_model(xdata, best_A_LTD)

    # compute r^2 values
    def r_squared(y_true, y_pred):
        residuals = y_true - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot)
    
    r_squared_LTP = r_squared(exp_LTP, ltp_model(pulse_num_LTP, best_A_LTP))
    r_squared_LTD = r_squared(exp_LTD, ltd_model(pulse_num_LTD, best_A_LTD))

    fig, ax = plt.subplots()
    ax.plot(pulse_num_LTP, exp_LTP, label='Exp. data (LTP)', ls='none', marker='o', color='b')
    ax.plot(pulse_num_LTD, exp_LTD, label='Exp. data (LTD)', ls='none', marker='o', color='r')
    ax.plot(xdata, y_bestfit_LTP, label=f'Best fit (LTP): A={best_A_LTP:.3f}, $R^2$={r_squared_LTP:.3f}', color='b')
    ax.plot(xdata, y_bestfit_LTD, label=f'Best fit (LTD): A={best_A_LTD:.3f}, $R^2$={r_squared_LTD:.3f}', color='r')
    ax.set(xlabel='Normalized Pulse Number', ylabel='Normalized Conductance')
    ax.legend()
    plt.draw()

    return best_A_LTP, best_A_LTD

if __name__ == '__main__':
    # read filename from command line
    if len(sys.argv) != 2:
        print("Usage: python nonlinear_fit.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    metadata, meas_params, meas_data = read_file(filename)

    for key, value in metadata.items():
        print(f"{key}: {value}")

    for key, value in meas_params.items():
        print(f"{key}: {value}")

    VStartPos = max(0.1, meas_params['startVolage1']) # (sic)
    VEndPos = min(4.1, meas_params['endVoltage1'])
    VStartNeg = min(0, meas_params['startVolage2']) # (sic)
    VEndNeg = max(-1.4, meas_params['endVoltage2'])
    twidth = meas_params['pulseWidth']

    kpos = np.where((meas_data[:,1] > VStartPos) & (meas_data[:,1] < VEndPos))
    kneg = np.where((meas_data[:,1] < VStartNeg) & (meas_data[:,1] > VEndNeg))

    plot_1(meas_data, kpos, kneg)

    # get on/off ratio
    onOffRatio = max(meas_data[kpos,3][0]) / min(meas_data[kpos,3][0])

    # write results to file
    allResults_filename = filename.replace('.csv', '_AllResults.dat')
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

    best_A_LTP, best_A_LTD = nonlinear_fit(pulse_num_LTP_norm, exp_LTP_norm, pulse_num_LTD_norm, exp_LTD_norm)

    print("Waiting for plots to close to terminate program...")
    plt.show()