import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    '''
    Chop a waveform into overlapping frames.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    frame_length (scalar) - length of the frame, in samples
    step (scalar) - step size, in samples
    
    @returns:
    frames (np.ndarray((num_frames, frame_length))) - waveform chopped into frames
       frames[m/step,n] = waveform[m+n] only for m = integer multiple of step
    '''
    # Ensure items are standard integers for sizing and strides
    frame_length = int(frame_length)
    step = int(step)
    
    # Calculate the total number of frames that fit completely inside the waveform boundary
    num_frames = int((len(waveform) - frame_length) // step) + 1
    
    # Calculate target memory stride patterns
    # itemsize tells us how many bytes a single scalar occupies (e.g. 8 bytes for float64)
    itemsize = waveform.itemsize
    
    # New strides definition: 
    # Row stride is the jump needed to hit the next step interval.
    # Column stride is the jump needed to hit the consecutive sample.
    strides = (step * itemsize, itemsize)
    
    # Create the matrix view over the existing waveform block memory
    frames = np.lib.stride_tricks.as_strided(
        waveform, 
        shape=(num_frames, frame_length), 
        strides=strides
    )
    
    return frames

def frames_to_mstft(frames):
    '''
    Take the magnitude FFT of every row of the frames matrix.
    
    @params:
    frames (np.ndarray((num_frames, frame_length))) - the speech samples
    
    @returns:
    mstft (np.ndarray((num_frames, frame_length))) - the magnitude short-time Fourier transform
    '''
    # Compute the FFT along rows (axis=-1) and grab its absolute magnitude
    mstft = np.abs(np.fft.fft(frames, axis=-1))
    return mstft

def mstft_to_spectrogram(mstft):
    '''
    Convert max(0.001*amax(mstft), mstft) to decibels.
    
    @params:
    stft (np.ndarray((num_frames, frame_length))) - magnitude short-time Fourier transform
    
    @returns:
    spectrogram (np.ndarray((num_frames, frame_length)) - spectrogram 
    
    The spectrogram should be expressed in decibels (20*log10(mstft)).
    np.amin(spectrogram) should be no smaller than np.amax(spectrogram)-60
    '''
    # Apply lower floor boundary based on max magnitude scaling
    floor_threshold = 0.001 * np.amax(mstft)
    bounded_mstft = np.maximum(floor_threshold, mstft)
    
    # Convert linear amplitude to log-scale decibels (20 * log10)
    spectrogram = 20 * np.log10(bounded_mstft)
    
    # Clamp the dynamic range floor to at most 60dB below the peak value
    db_floor = np.amax(spectrogram) - 60
    spectrogram = np.maximum(spectrogram, db_floor)
    
    return spectrogram