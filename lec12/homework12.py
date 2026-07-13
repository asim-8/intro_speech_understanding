import numpy as np
import scipy.signal

def voiced_excitation(duration, F0, Fs):
    # Initialize a silent excitation vector array
    excitation = np.zeros(int(duration))
    # Calculate fundamental pitch period interval in samples
    period = int(np.round(Fs / F0))
    # Assign the glottal impulses to every integer multiple index
    excitation[::period] = -1
    return excitation

def resonator(x, F, BW, Fs):
    # Calculate exact parameters as defined by the test script
    C = -np.exp(-2 * np.pi * BW / Fs)
    B = 2 * np.exp(-np.pi * BW / Fs) * np.cos(2 * np.pi * F / Fs)
    A = 1 - B - C
    
    # Map matching coefficients to scipy.signal.lfilter:
    # y[n] = A*x[n] + B*y[n-1] + C*y[n-2] -> b=[A], a=[1, -B, -C]
    b = [A]
    a = [1.0, -B, -C]
    
    return scipy.signal.lfilter(b, a, x)

def synthesize_vowel(duration, F0, F1, F2, F3, F4, BW1, BW2, BW3, BW4, Fs):
    excitation = voiced_excitation(duration, F0, Fs)
    y1 = resonator(excitation, F1, BW1, Fs)
    y2 = resonator(y1, F2, BW2, Fs)
    y3 = resonator(y2, F3, BW3, Fs)
    speech = resonator(y3, F4, BW4, Fs)
    return speech