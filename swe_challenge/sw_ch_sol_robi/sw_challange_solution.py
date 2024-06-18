
# Lorentzian fit
from lmfit.models import LorentzianModel, ConstantModel
import numpy as np
import yaml
import plotly.graph_objects as go

# New imports
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Peak detection
from scipy.signal import find_peaks as fp


# Multi-threading
from multiprocessing import Array, Process, Lock

##########################################################################

# Variables
x_dim = 50
y_dim = 50

nr_of_peaks = 6

##########################################################################
# FUNCTIONS

# 2dplot:


def linep2dp(
        x: np.ndarray,
        y: list | np.ndarray,
        yn="",
        title="",
        xaxis_title="x",
        yaxis_title="y") -> None:
    """Multi-line 2D plot using plotly.

    :param x: x-axis values.
    :param yv: y-axis values.
    :param yn: y-axis names.
    :param title: Title of the plot.
    :param xaxis_title: Title of the x-axis.
    :param yaxis_title: Title of the y-axis.
    """

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=yn))
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title)
    fig.show()

# Filter peaks based on difference:
# From optimization dleta_tmp is optimal for value of 80 data point difference


def filter_peaks(peak_exceptions, delta_tmp):
    list_tmp = np.ndarray.tolist(peak_exceptions)
    j = 0
    while j < len(list_tmp) - 1:
        if np.abs(list_tmp[j] - list_tmp[j + 1]) < delta_tmp:
            list_tmp.pop(j + 1)
        else:
            j = j + 1
    return (np.array(list_tmp))

# Calculate exceptions based on the neighboring values


def average_on_neighbours(peaks):
    for i in range(x_dim):
        for j in range(y_dim):
            if (peaks[i][j][0] == 0):
                sum = np.zeros((nr_of_peaks))
                nr_of_neighbours = np.zeros((nr_of_peaks))

                for k in range(nr_of_peaks):
                    # left neighbour
                    if (i > 0 and i < x_dim):
                        sum[k] = sum[k] + peaks[i - 1][j][k]
                        nr_of_neighbours[k] = nr_of_neighbours[k] + 1
                    # right neighbour
                    if (i >= 0 and i < (x_dim - 1)):
                        sum[k] = sum[k] + peaks[i + 1][j][k]
                        nr_of_neighbours[k] = nr_of_neighbours[k] + 1
                    # top neighbour
                    if (j > 0 and j < y_dim):
                        sum[k] = sum[k] + peaks[i][j - 1][k]
                        nr_of_neighbours[k] = nr_of_neighbours[k] + 1
                    # bottom neighbour
                    if (j >= 0 and j < (y_dim - 1)):
                        sum[k] = sum[k] + peaks[i][j + 1][k]
                        nr_of_neighbours[k] = nr_of_neighbours[k] + 1

                    peaks[i][j][k] = (int)(sum[k] / nr_of_neighbours[k])
    return peaks

##########################################################################
# ALGORITHM


# Found values for good approximation and filterings
best_prominance = 0.0004
best_width = 1e-5
best_window_legth = 260
best_polyorder = 6
best_delta = 80


def find_peak_pos(x_data_tmp, X_source):

    # peak_positions =
    peak_positions = np.zeros((x_dim, y_dim, nr_of_peaks))
    # For exceptions
    peak_exceptions = []  # format: [i,j,(peaks)]

    for i in range(0, x_dim):
        for j in range(0, y_dim):
            y_data = X_source[:, i, j]

            # Savitzky-Golay filtering
            y_data_continuous = savgol_filter(
                y_data,
                window_length=best_window_legth,
                polyorder=best_polyorder,
                mode="nearest")
            # Peak finder from scipy
            y_data_continuous = -1 * y_data_continuous
            peaks, properties = fp(
                y_data_continuous, prominence=best_prominance, width=best_width)
            y_data_continuous = -1 * y_data_continuous

            # Peak filtering based on minimal distance
            peaks = filter_peaks(peaks, best_delta)
            if len(peaks) == nr_of_peaks:
                peak_positions[i][j][:] = peaks
            else:
                peak_exceptions.append((i, j, peaks))
    # Average exceptions
    peak_positions = average_on_neighbours(peak_positions)

    # Show exceptions after averaging
    for i in range(0, x_dim):
        for j in range(0, y_dim):
            for k in range(len(peak_exceptions)):
                y_data = X_source[:, i, j]
                if (i == peak_exceptions[k][0] and j == peak_exceptions[k][1]):
                    peaks = peak_positions[i][j][:]
                    # Savitzky-Golay filtering
                    y_data_continuous = savgol_filter(
                        y_data,
                        window_length=best_window_legth,
                        polyorder=best_polyorder,
                        mode="nearest")
    return (peak_positions, peak_exceptions)


def calc_B(peak_positions_tmp, x_data_tmp):
    B_tmp = np.zeros((x_dim, y_dim, 3))
    x_array = np.array(x_data_tmp)
    for i in range(x_dim):
        for j in range(y_dim):
            B_tmp[i][j][0] = np.abs(x_array[(int)(
                peak_positions_tmp[i][j][0])] - x_array[(int)(peak_positions_tmp[i][j][5])])
            B_tmp[i][j][1] = np.abs(x_array[(int)(
                peak_positions_tmp[i][j][1])] - x_array[(int)(peak_positions_tmp[i][j][4])])
            B_tmp[i][j][2] = np.abs(x_array[(int)(
                peak_positions_tmp[i][j][2])] - x_array[(int)(peak_positions_tmp[i][j][3])])
    return B_tmp

# VARIABLES FOR LORENTZIAN FIT


delta_peak_freq = 0.032  # Ghz

amp = 0.00015
min_amp = 0.00001
max_amp = 0.0002

sigma = 0.0006
min_sigma = 0.0002
max_sigma = 0.0009


def multi_lorentz_fit(peak_pos, i, j, k, x_tmp, y_tmp, peaks_freq):

    # Finding constant bachground
    model = ConstantModel()
    params = model.make_params()
    params['c'].set(-1, min=-1.01, max=-0.99)

    # Tripple-Lorentzian model
    l1 = LorentzianModel(prefix='l1_')
    l2 = LorentzianModel(prefix='l2_')
    l3 = LorentzianModel(prefix='l3_')

    # Initial guess
    p1 = l1.make_params()
    p1['l1_center'].set(
        peaks_freq[0],
        min=peaks_freq[0] -
        delta_peak_freq,
        max=peaks_freq[0] +
        delta_peak_freq)
    p1['l1_amplitude'].set(amp, min=min_amp, max=max_amp)
    p1['l1_sigma'].set(sigma, min=min_sigma, max=max_sigma)

    p2 = l2.make_params()
    p2['l2_center'].set(
        peaks_freq[1],
        min=peaks_freq[1] -
        delta_peak_freq,
        max=peaks_freq[1] +
        delta_peak_freq)
    p2['l2_amplitude'].set(amp, min=min_amp, max=max_amp)
    p2['l2_sigma'].set(sigma, min=min_sigma, max=max_sigma)

    p3 = l3.make_params()
    p3['l3_center'].set(
        peaks_freq[2],
        min=peaks_freq[2] -
        delta_peak_freq,
        max=peaks_freq[2] +
        delta_peak_freq)
    p3['l3_amplitude'].set(amp, min=min_amp, max=max_amp)
    p3['l3_sigma'].set(sigma, min=min_sigma, max=max_sigma)

    model = model + l1 + l2 + l3
    params.update(p1)
    params.update(p2)
    params.update(p3)

    # print(model)
    # print(params)
    # plt.figure()
    # init=model.eval(params=params,x=x_tmp)
    # plt.plot(x_tmp,-y_tmp)
    # plt.plot(x_tmp,init)

    result = model.fit(data=y_tmp, params=params, x=x_tmp)
    # comps=result.eval_components()
    # plt.plot(x_tmp, -result.best_fit)
    peak_pos[i][j][k] = (result.params["l1_center"].value +
                         result.params["l2_center"].value + result.params["l3_center"].value) / 3

    # print(result.params["l1_center"].value, result.params["l2_center"].value, result.params["l3_center"].value)
    # plt.show()


def multi_segment_fit(
        B_array,
        progress,
        lock,
        x_range,
        y_range,
        rough_estimate_peak_positions,
        x_data,
        X):

    peak_positions = np.zeros((x_dim, y_dim, nr_of_peaks))
    x_resolution = (x_data[len(x_data) - 1] - x_data[0]) / len(x_data)
    deltaX = 0.002  # GHz, distance from middle peak, rough estimation for initial guess
    deltaX_dat_points = (int)(deltaX / x_resolution)

    for i in x_range:
        for j in y_range:
            if (i >= 0 and i < x_dim and j >= 0 and j < y_dim):
                segments_data_point = np.zeros((8))
                # print(i,j)
                y_data = X[:, i, j]
                # Calculating segments:
                for k in range(len(rough_estimate_peak_positions[i][j])):
                    if k < (len(rough_estimate_peak_positions[i][j]) - 1):
                        segments_data_point[k +
                                            1] = (int)((rough_estimate_peak_positions[i][j][k] +
                                                        rough_estimate_peak_positions[i][j][k +
                                                                                            1]) /
                                                       2)
                    elif k == (len(rough_estimate_peak_positions[0][0]) - 1):
                        segments_data_point[k + 1] = (int)(
                            (rough_estimate_peak_positions[i][j][k] + len(x_data)) / 2)
                # add last element:
                segments_data_point[7] = (int)(len(x_data))
                # Lorentzian fit for each segment
                for k in range(len(rough_estimate_peak_positions[i][j])):
                    x0 = (int)(segments_data_point[k])
                    x1 = (int)(segments_data_point[k + 1])

                    x_values = x_data[x0:x1]
                    # x_values_tmp = [k-x0 for k in x_values]

                    # y_data preparation
                    y_values = y_data[x0:x1]
                    y_values = savgol_filter(
                        y_values, window_length=10, polyorder=3, mode="nearest")

                    x = x_values
                    y = -y_values

                    rough_peak_pos_freq = (x_data[(int)(rough_estimate_peak_positions[i][j][k] -
                                                        deltaX_dat_points) if (rough_estimate_peak_positions[i][j][k] -
                                                                               deltaX_dat_points) > 0 else 0], x_data[(int)(rough_estimate_peak_positions[i][j][k])], x_data[(int)(rough_estimate_peak_positions[i][j][k] +
                                                                                                                                                                                   deltaX_dat_points if (rough_estimate_peak_positions[i][j][k] +
                                                                                                                                                                                                         deltaX_dat_points) < len(x_data) else len(x_data) -
                                                                                                                                                                                   1)])
                    multi_lorentz_fit(peak_positions, i, j, k, np.array(
                        x), np.array(y), np.array(rough_peak_pos_freq))
                B_array[(i * x_dim + j) * 3 + 0] = np.abs(peak_positions[i]
                                                          [j][0] - peak_positions[i][j][5])
                B_array[(i * x_dim + j) * 3 + 1] = np.abs(peak_positions[i]
                                                          [j][1] - peak_positions[i][j][4])
                B_array[(i * x_dim + j) * 3 + 2] = np.abs(peak_positions[i]
                                                          [j][2] - peak_positions[i][j][3])
                # print(B_array[(i*x_dim + j)*3 + 0], B_array[(i*x_dim + j)*3 + 1], B_array[(i*x_dim + j)*3 + 2])
                lock.acquire()
                progress[0] = progress[0] + 1
                lock.release()

##########################################################################
# MAIN WITH MULTI THREADING


def show_progress(progress, lock):  # n is the dedicated location in the array
    lock.acquire()
    progreess_tmp = progress[0] / (x_dim * y_dim)
    lock.release()
    while progreess_tmp != 1:
        lock.acquire()
        progreess_tmp = progress[0] / (x_dim * y_dim)
        lock.release()
        print("\tMultiprocess: ", progreess_tmp * 100, "% \r", end="\033[K")


def precise_calc(rough_estimate_peak_positions, x, y):

    nr_of_processes = 10
    # nr_of_processes = 1
    processing_range = (int)(x_dim / nr_of_processes)

    B = np.zeros((x_dim * y_dim * 3))
    # no need for lock, each process will access different part of the array
    B_array = Array('f', B)

    progress = Array('i', [0])
    progress_lock = Lock()
    multi_process = []

    # Progress bar
    progress_show_process = Process(
        target=show_progress, args=(
            progress, progress_lock))
    progress_show_process.start()

    # Start processes
    for i in range(nr_of_processes):
        # print(i)
        multi_process.append(
            Process(
                target=multi_segment_fit,
                args=(
                    B_array,
                    progress,
                    progress_lock,
                    np.arange(
                        i * processing_range,
                        (i + 1) * processing_range),
                    np.arange(
                        0,
                        y_dim),
                    rough_estimate_peak_positions,
                    x,
                    y,
                )))
        # multi_process.append(Process(target=multi_segment_fit, args=(B_array, progress, progress_lock, np.arange(4,5), np.arange(3, 4), rough_estimate_peak_positions, x, y, )))

    print("\tMultiprocess: multi process prepared")
    for i in multi_process:
        i.start()
    print("\tMultiprocess: multi process started")
    for i in multi_process:
        i.join()

    progress_lock.acquire()
    progress_tmp = progress[0]
    progress[0] = x_dim * y_dim
    progress_lock.release()

    progress_show_process.join()
    B = B_array[:]

    print("\tMultiprocess: ended at: ",
          progress_tmp / (x_dim * y_dim) * 100, "%")

    return B


def rough_calc(x, y):

    # Rough estimate, based only on current peak positions
    B_rough_estimate = np.zeros((x_dim, y_dim, 3))
    rough_estimate_peak_positions, peak_exceptions = find_peak_pos(x, y)
    B_rough_estimate = calc_B(rough_estimate_peak_positions, x)
    # print("B_rough_estimate[12,21]:",B_rough_estimate[12][21][:])
    # print((B_rough_estimate[:,:,0]))
    return (rough_estimate_peak_positions, B_rough_estimate)


if __name__ == "__main__":

    # Loading data

    # Without magnetic field
    X = np.load("ESR_Continuous_2024-03-07-17-46-03_PCB_ref_50x50.npy")
    X = np.sum(X[0] / X[1], axis=1)
    with open("ESR_Continuous_2024-03-07-17-46-03_PCB_ref_50x50.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        frq = cfg["frequency_values"]  # frequency values in Hz
    # print(f"X: {X.shape}")
    # print(f"frq: {len(frq)}")
    x_data = [f * 1e-9 for f in frq]

    # With magnetic field
    X_mag = np.load(
        "ESR_Continuous_2024-03-07-17-58-48_PCB_Top_25mA_50x50.npy")
    X_mag = np.sum(X_mag[0] / X_mag[1], axis=1)
    with open("ESR_Continuous_2024-03-07-17-58-48_PCB_Top_25mA_50x50.yaml", "r") as f_mag:
        cfg_mag = yaml.safe_load(f_mag)
        frq_mag = cfg_mag["frequency_values"]  # frequency values in Hz
    # print(f"X_mag: {X_mag.shape}")
    # print(f"frq: {len(frq_mag)}")
    x_mag_data = [f_mag * 1e-9 for f_mag in frq_mag]

    print("Data loaded!")

    # # Rough estimation of peak positions
    # #	rough_estimate_peak_position has 50x50x6 elements, 6 position of 6 peaks for each data-set
    # #	B has 50x50x3 elements, the 3 peak difference calc for B vector of 50x50 data_set
    # #	3 values = (|P6-P1| , |P5-P2|, |P4-P3|)
    rough_estimate_peak_positions, B_rough = rough_calc(x_data, X)
    np.save("processing_reults/B_rough", B_rough)
    print("B rough estimate done!")

    rough_estimate_peak_positions_mag, B_mag_rough = rough_calc(
        x_mag_data, X_mag)
    np.save("processing_reults/B_mag_rough", B_rough)
    print("B_mag rough estimate done!")

    # # Precise calculations
    # #	B has 7500 elements, 3x50x50, 3 calc B values, following the second data-set's 3 calculated values
    # #	3 values = (|P6-P1| , |P5-P2|, |P4-P3|)
    B = precise_calc(rough_estimate_peak_positions, x_data, X)
    np.save("processing_reults/B_Lorentzian", B)
    print("B with 3-Lorentzian estimate done!")

    B_mag = precise_calc(rough_estimate_peak_positions, x_mag_data, X_mag)
    np.save("processing_reults/B_mag_Lorentzian", B_mag)
    print("B_mag with 3-Lorentzian estimate done!")

    print("Done")
