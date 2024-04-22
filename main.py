import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft, rfftfreq
import pywt
def gaussian(x, x0, sigma):
    return np.exp(-np.power((x - x0) / sigma, 2.0) / 2.0)


def make_chirp(t, t0, a):
    frequency = (a * (t + t0)) ** 2
    frequency = 1;
    chirp = np.sin(2 * np.pi * frequency * t)
    return chirp, frequency

# generate signal
time = np.linspace(0, 1, 2000)

some = np.cos(time * 20 * np.pi) * gaussian(time, 0.5, 1) + np.sin(20 * time) * gaussian(time, 0.5, 1) + np.cos(time * 60 * np.pi) * gaussian(time, 0.5, 10)
# perform CWT
wavelet = "gaus6"
# logarithmic scale for scales, as suggested by Torrence & Compo:
widths = np.geomspace(1, 1024, num=100)
sampling_period = np.diff(time).mean()
cwtmatr, freqs = pywt.cwt(some, widths, wavelet, sampling_period=sampling_period)
# absolute take absolute value of complex result
cwtmatr = np.abs(cwtmatr[:-1, :-1])

# plot result using matplotlib's pcolormesh (image with annoted axes)
fig, axs = plt.subplots(2, 1)
pcm = axs[0].pcolormesh(time, freqs, cwtmatr)
axs[0].set_yscale("log")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Frequency (Hz)")
axs[0].set_title("Continuous Wavelet Transform (Scaleogram)")
fig.colorbar(pcm, ax=axs[0])

# plot fourier transform for comparison

yf = rfft(some)
xf = rfftfreq(len(some), sampling_period)
plt.semilogx(xf, np.abs(yf))
axs[1].set_xlabel("Frequency (Hz)")
axs[1].set_title("Fourier Transform")
plt.tight_layout()

plt.show()

