import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f, with sampling frequency Fs.

    @param:
    f (scalar): frequency of the root tone, in Hertz
    Fs (scalar): sampling frequency, in samples/second

    @return:
    x (array): a one-half-second waveform containing the chord
    '''
    # Calculate duration in samples (0.5 seconds)
    N = int(0.5 * Fs)
    n = np.arange(N)
    
    # Calculate frequencies based on equal temperament semitones
    f_root = f
    f_third = f * (2 ** (4 / 12))
    f_fifth = f * (2 ** (7 / 12))
    
    # Convert to radial frequencies (omega)
    omega_root = 2 * np.pi * f_root / Fs
    omega_third = 2 * np.pi * f_third / Fs
    omega_fifth = 2 * np.pi * f_fifth / Fs
    
    # Generate and sum the three pure tones
    x = np.cos(omega_root * n) + np.cos(omega_third * n) + np.cos(omega_fifth * n)
    return x

def dft_matrix(N):
    '''
    Create a DFT transform matrix, W, of size N.
    
    @param:
    N (scalar): number of columns in the transform matrix
    
    @result:
    W (NxN array): a matrix of dtype='complex' whose (k,n)^th element is:
           W[k,n] = cos(2*np.pi*k*n/N) - j*sin(2*np.pi*k*n/N)
    '''
    # Create 1D arrays for rows (k) and columns (n)
    k = np.arange(N)
    default_n = np.arange(N)
    
    # Use broadcasting to create a 2D matrix of (k * n) indices
    kn = k[:, None] * default_n[None, :]
    
    # Calculate the exponential matrix components
    W = np.cos(2 * np.pi * kn / N) - 1j * np.sin(2 * np.pi * kn / N)
    return W

def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.

    @param:
    x (array): the waveform
    Fs (scalar): sampling frequency (samples/second)

    @return:
    f1, f2, f3: The three loudest frequencies (in Hertz)
      These should be sorted so f1 < f2 < f3.
    '''
    N = len(x)
    
    # Compute the discrete Fourier transform using the DFT matrix
    W = dft_matrix(N)
    X = W @ x
    
    # Isolate the positive frequency spectrum (up to Nyquist limit)
    half_N = N // 2
    X_positive = X[:half_N]
    magnitudes = np.abs(X_positive)
    
    # Find indices of the 3 highest peak magnitudes
    # argsort sorts ascending, so [-3:] grabs the 3 largest elements
    top_3_indices = np.argsort(magnitudes)[-3:]
    
    # Convert DFT indices back to physical frequencies: f = k * Fs / N
    loudest_frequencies = top_3_indices * Fs / N
    
    # Ensure they are cleanly sorted ascending (f1 < f2 < f3)
    loudest_frequencies.sort()
    
    return loudest_frequencies[0], loudest_frequencies[1], loudest_frequencies[2]