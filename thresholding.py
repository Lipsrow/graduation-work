import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec, waverec, dwt_max_level
from scipy.stats import norm
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from copy import deepcopy
import soundfile as sf
import os

np.random.seed(42)
PATH = ''
file = ''
audio_file = ''
db_levels = list(map(str, range(1, 39)))
coif_levels = list(map(str, range(1, 18)))
sym_levels = list(map(str, range(2, 21)))
st_dict = {'db': db_levels, 'coif': coif_levels, 'sym': sym_levels}
a = 0
b = 1


def F(x, sigma, T, mode):
    if abs(x) <= T ** 2:
        return x - sigma ** 2
    if mode == 'soft':
        return sigma ** 2 + T ** 2
    else:
        return sigma ** 2


def get_risk(signal, sigma, T, mode):
    risk = 0
    for x in signal:
        risk += F(x ** 2, sigma, T, mode)
    return risk


def norm_X(X):
    return np.sqrt(np.sum(X ** 2))


def stable(signal):
    return np.std(signal) * np.sqrt(len(signal) / (len(signal) - 1))


def MAD(signal):
    return (np.median(abs(signal - np.median(signal)))) / 0.6745


def IQR(signal):
    return (np.percentile(signal, 75) - np.percentile(signal, 25)) / 1.349


def SURE_level(coeffs, sigma, X):
    N = len(coeffs)
    e = sigma ** 2 * np.sqrt(N) * np.log(N) ** (1.5)
    T_U = sigma * np.sqrt(2 * np.log(N))
    if X - N * sigma <= e:
        T = T_U
    else:
        T = 0
        min_risk = np.inf
        signal_sorted = np.sort(abs(coeffs))[::-1]
        for l in range(len(signal_sorted)):
            risk = np.sum(signal_sorted[l:] ** 2) - (N - l) * sigma ** 2 + l * (sigma ** 2 + signal_sorted[l])
            if risk < min_risk:
                T = signal_sorted[l]
                min_risk = risk
    return T


def FDR_level(coeffs, sigma, q):
    N = len(coeffs)

    def p_value(x):
        return 2 * (1 - norm.cdf(abs(x) / sigma))

    P = p_value(coeffs)
    P = np.sort(P)
    k = 0
    for i in range(N - 1, -1, -1):
        if P[i] <= i * q / N:
            k = i
            break
    T = sigma * norm.ppf(1 - P[k] / 2)
    return T


def G(coeffs, T):
    coeffs_thresh = deepcopy(coeffs)
    mu = 0
    coeffs_thresh[abs(coeffs_thresh) <= T] = 0
    coeffs_thresh[coeffs_thresh < -T] += T
    coeffs_thresh[coeffs_thresh > T] -= T
    mu += len(coeffs_thresh[coeffs_thresh == 0])
    mu /= len(coeffs)
    s = np.sum(np.square(coeffs - coeffs_thresh))
    if mu:
        return s / mu ** 2
    return np.inf


def GCV_level(coeffs):
    a = np.min(abs(coeffs)) * 1.001
    b = np.max(abs(coeffs)) * 0.999
    n = 30
    fib_sequence = [1, 1]
    for i in range(2, n + 1):
        fib_sequence.append(fib_sequence[i - 1] + fib_sequence[i - 2])

    x1 = a + (b - a) * (fib_sequence[n - 2] / fib_sequence[n])
    x2 = a + (b - a) * (fib_sequence[n - 1] / fib_sequence[n])
    for i in range(1, n - 1):
        if G(coeffs, x1) > G(coeffs, x2):
            a = x1
            x1 = x2
            x2 = a + (b - a) * (fib_sequence[n - i - 1] / fib_sequence[n - i])
        else:
            b = x2
            x2 = x1
            x1 = a + (b - a) * (fib_sequence[n - i - 2] / fib_sequence[n - i])
    return (a + b) / 2


def FDR_thresholding(signal, level, st, std, mode, q):
    coefficients = wavedec(signal, st, level=level)
    sigma = std(coefficients[-1])
    risk = 0
    for i in range(1, len(coefficients)):
        T = FDR_level(coefficients[i], sigma, q)
        risk += get_risk(coefficients[i], sigma, T, mode)
        coefficients[i][abs(coefficients[i]) <= T] = 0
    signal_thresh = waverec(coefficients, st)
    return signal_thresh, sigma, risk


def SURE_thresholding(signal, level, st, std, mode):
    X = norm_X(signal)
    coefficients = wavedec(signal, st, level=level)
    sigma = std(coefficients[-1])
    risk = 0
    for i in range(1, len(coefficients)):
        T = SURE_level(coefficients[i], sigma, X)
        risk += get_risk(coefficients[i], sigma, T, mode)
        coefficients[i][abs(coefficients[i]) <= T] = 0
    signal_thresh = waverec(coefficients, st)
    return signal_thresh, sigma, risk


def GCV_thresholding(signal, level, st, std):
    coefficients = wavedec(signal, st, level=level)
    risk = 0
    sigma = std(coefficients[-1])
    for i in range(1, len(coefficients)):
        T = GCV_level(coefficients[i])
        risk += get_risk(coefficients[i], sigma, T, 'soft')
        coefficients[i][abs(coefficients[i]) <= T] = 0
    signal_thresh = waverec(coefficients, st)
    return signal_thresh, sigma, risk


def plot(a, b, signal, signal_thresh, sigma_calc, risk):
    fig, axes = plt.subplots(2)
    x1 = np.linspace(a, b, len(signal))
    x2 = np.linspace(a, b, len(signal_thresh))
    axes[0].plot(x1, signal)
    axes[1].plot(x2, signal_thresh)
    plt.suptitle(f"Оценка СКО: {sigma_calc}, Оценка риска: {risk}")
    axes[0].set_title("Сигнал до обработки")
    axes[1].set_title("Сигнал после обработки")
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    plt.show()


def get_best(method, signal, mode, std, q=0.001):
    min_risk = np.inf
    best_st = ''
    st1 = 'db'
    st2 = 'coif'
    st3 = 'sym'
    for i in range(1, 39):
        level = dwt_max_level(len(signal), st1 + str(i))
        if method == SURE_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st1 + str(i), std, mode)
        elif method == GCV_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st1 + str(i), std)
        else:
            signal_thresh, sigma_calc, risk = method(signal, level, st1 + str(i), std, mode, q)
        if 0 < risk < min_risk:
            min_risk = risk
            best_st = st1 + str(i)
    for i in range(1, 18):
        level = dwt_max_level(len(signal), st2 + str(i))
        if method == SURE_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st2 + str(i), std, mode)
        elif method == GCV_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st2 + str(i), std)
        else:
            signal_thresh, sigma_calc, risk = method(signal, level, st2 + str(i), std, mode, q)
        if 0 < risk < min_risk:
            min_risk = risk
            best_st = st2 + str(i)

    for i in range(2, 21):
        level = dwt_max_level(len(signal), st3 + str(i))
        if method == SURE_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st3 + str(i), std, mode)
        elif method == GCV_thresholding:
            signal_thresh, sigma_calc, risk = method(signal, level, st3 + str(i), std)
        else:
            signal_thresh, sigma_calc, risk = method(signal, level, st3 + str(i), std, mode, q)
        if 0 < risk < min_risk:
            min_risk = risk
            best_st = st3 + str(i)

    level = dwt_max_level(len(signal), best_st)
    if method == SURE_thresholding:
        signal_thresh, sigma_calc, risk = method(signal, level, best_st, std, mode)
    elif method == GCV_thresholding:
        signal_thresh, sigma_calc, risk = method(signal, level, best_st, std)
    else:
        signal_thresh, sigma_calc, risk = method(signal, level, best_st, std, mode, q)
    return signal_thresh, sigma_calc, risk


######################
def open_file():
    global file
    file = filedialog.askopenfilename()

def open_audio():
    global audio_file
    audio_file = filedialog.askopenfilename()


def save():
    global PATH
    PATH = filedialog.askdirectory()


def show_q(event):
    method = combo_method.get()
    if method == 'FDR':
        q_label.config(state='normal')
        q_entry.config(state='normal')
    else:
        q_label.config(state='disabled')
        q_entry.config(state='disabled')


def get_audio_filename(orig_filename, PATH):
    if not PATH:
        ind = orig_filename.rfind(".")
        filename = orig_filename[:ind] + " processed"
        if not os.path.exists(filename + ".wav"):
            return filename + ".wav"
        number = 1
        while os.path.exists(filename + str(number) + ".wav"):
            number += 1
        return filename + str(number) + ".wav"
    else:
        r = orig_filename.rfind(".")
        l = orig_filename.rfind("/")
        filename = orig_filename[l + 1:r] + " processed"
        if not os.path.exists(PATH + "/" + filename + ".wav"):
            return PATH + "/"+ filename + ".wav"
        number = 1
        while os.path.exists(PATH + "/"+ filename + str(number) + ".wav"):
            number += 1
        return PATH + "/"+ filename + str(number) + ".wav"

def get_npy_filename(orig_filename, PATH):
    if not PATH:
        ind = orig_filename.rfind(".")
        filename = orig_filename[:ind] + " processed"
        if not os.path.exists(filename + ".npy"):
            return filename + ".npy"
        number = 1
        while os.path.exists(filename + str(number) + ".npy"):
            number += 1
        return filename + str(number) + ".npy"
    else:
        r = orig_filename.rfind(".")
        l = orig_filename.rfind("/")
        filename = orig_filename[l + 1:r] + " processed"
        if not os.path.exists(PATH + "/" + filename + ".npy"):
            return PATH + "/" + filename + ".npy"
        number = 1
        while os.path.exists(PATH + "/" + filename + str(number) + ".npy"):
            number += 1
        return PATH + "/" + filename + str(number) + ".npy"

def change_pow(event):
    wavelet = combo_wavelet.get()
    if wavelet == 'Добеши':
        combo_pow.config(values=(db_levels))
    elif wavelet == 'Coiflet':
        combo_pow.config(values=(coif_levels))
    elif wavelet == 'Symlet':
        combo_pow.config(values=(sym_levels))


def main():
    method = combo_method.get()
    mode = combo_mode.get()
    if mode == 'Мягкая':
        mode = 'soft'
    elif mode == 'Жесткая':
        mode = 'hard'
    st = combo_wavelet.get()
    if st == "Symlet":
        st = 'sym'
    elif st == 'Добеши':
        st = 'db'
    elif st == 'Coiflet':
        st = 'coif'
    std = combo_std.get()
    if file:
        signal = np.load(file)
        N = len(signal)
    elif audio_file:
        signal, sample_rate = sf.read(audio_file)
        N = len(signal)
    else:
        raise ValueError("Не выбран файл")
    if std == 'Выборочная':
        std = stable
    elif std == 'IQR':
        std = IQR
    elif std == 'MAD':
        std = MAD
    else:
        pass
    if method == "FDR":
        q = float(q_entry.get())
        st += combo_pow.get()
        level = dwt_max_level(N, st)
        signal_thresh, sigma_calc, risk = FDR_thresholding(signal, level, st, std, mode, q)
    elif method == "SURE":
        st += combo_pow.get()
        level = dwt_max_level(N, st)
        signal_thresh, sigma_calc, risk = SURE_thresholding(signal, level, st, std, mode)
    elif method == "GCV":
        st += combo_pow.get()
        level = dwt_max_level(N, st)
        signal_thresh, sigma_calc, risk = GCV_thresholding(signal, level, st, std)
    else:
        raise ValueError("Не выбран метод!")
    if file:
        path_to_save = get_npy_filename(file, PATH)
        np.save(path_to_save, signal_thresh)
    elif audio_file:
        path_to_save = get_audio_filename(audio_file, PATH)
        sf.write(path_to_save, signal_thresh, sample_rate)
    plot(a, b, signal, signal_thresh, sigma_calc, risk)


FONT = ("Times New Roman", 17)
SMALL_FONT = ("Times New Roman", 15, "italic")
root = tk.Tk()
root.option_add("*TCombobox*Listbox.font", SMALL_FONT)
root.geometry("1300x400+125+150")
root.title("Обработка сигналов")

frame_load = tk.Frame()
tk.Label(frame_load, text="Выберите файл со значениями сигнала\n(с расширением .npy)", font=FONT).grid(row=0, column=0)
btn_file = tk.Button(frame_load, text="Выбрать файл", command=open_file, font=SMALL_FONT)
btn_file.grid(row=1, column=0)
frame_load.grid(row=0, column=0, padx=10)

frame_audio = tk.Frame()
tk.Label(frame_audio, text="Выберите аудио для обработки\n(с расширением .wav)", font=FONT).grid(row=0, column=0)
btn_file = tk.Button(frame_audio, text="Выбрать файл", command=open_audio, font=SMALL_FONT)
btn_file.grid(row=1, column=0)
tk.Label(frame_audio, text="Выберите путь для сохранения", font=FONT).grid(row=0, column=1, padx=30)
btn_save = tk.Button(frame_audio, text="Выбрать файл", command=save, font=SMALL_FONT)
btn_save.grid(row=1, column=1, padx=30)
frame_audio.grid(row=0, column=1, padx=10)

frame_method = tk.Frame(root)
q_label = tk.Label(frame_method, text='q = ', font=FONT, state='normal')
q_entry = tk.Entry(frame_method, state='normal', font=SMALL_FONT)
q_entry.insert(0, "0.001")
q_entry.grid(row=1, column=1, sticky='w')
q_label.grid(row=1, column=0, sticky='e')
tk.Label(frame_method, text='Выберите метод обработки', font=FONT).grid(row=0, column=0)
combo_method = ttk.Combobox(frame_method, values=['SURE', 'GCV', 'FDR'], font=SMALL_FONT)
combo_method.set("FDR")
combo_method.bind("<<ComboboxSelected>>", show_q)
combo_method.grid(row=0, column=1)
frame_method.grid(row=1, column=0, pady=20, padx=10)

frame_mode = tk.Frame(root)
tk.Label(frame_mode, text="Выберите режим обработки", font=FONT).grid(row=0, column=0, padx=20)
combo_mode = ttk.Combobox(frame_mode, font=SMALL_FONT, values=['Мягкая', 'Жесткая'], width=26)
combo_mode.set("Мягкая")
combo_mode.grid(row=1, column=0)
tk.Label(frame_mode, text="Выберите способ оценки дисперсии", font=FONT).grid(row=0, column=1, padx=40)
combo_std = ttk.Combobox(frame_mode, font=SMALL_FONT, values=['MAD', 'IQR', 'Выборочная'], width=30)
combo_std.grid(row=1, column=1)
combo_std.set('Выборочная')
frame_mode.grid(row=1, column=1, sticky='w', pady=20, padx=10)

frame_wavelet = tk.Frame(root)
tk.Label(frame_wavelet, text="Выберите вейвлет", font=FONT).grid(row=0, column=0, columnspan=2, pady=10)
tk.Label(frame_wavelet, text="Название вейвлета", font=FONT).grid(row=1, column=0, padx=10)
combo_wavelet = ttk.Combobox(frame_wavelet, font=SMALL_FONT,
                             values=['Coiflet', 'Добеши', 'Symlet'])
combo_wavelet.set("Добеши")
combo_wavelet.grid(row=1, column=1)
combo_wavelet.bind("<<ComboboxSelected>>", change_pow)
tk.Label(frame_wavelet, text="Порядок вейвлета", font=FONT).grid(row=2, column=0, padx=30, sticky='e')
combo_pow = ttk.Combobox(frame_wavelet, font=SMALL_FONT,
                         values=list(map(str, range(1, 39))))
combo_pow.set('12')
combo_pow.grid(row=2, column=1, sticky='we')
frame_wavelet.grid(row=2, column=0, pady=20, padx=10, sticky='we')

btn_frame = tk.Frame(root, highlightbackground="grey",
                         highlightthickness=1, bd=0)
btn_start = tk.Button(btn_frame, text="Выполнить", font=SMALL_FONT, command=main, width=25, height=4)
btn_start.grid(row=0, column=0)
btn_frame.grid(row=2, column=1, columnspan=2, pady=10)
root.mainloop()
