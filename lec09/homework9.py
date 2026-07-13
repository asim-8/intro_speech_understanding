import numpy as np

def VAD(waveform, Fs):
    '''
    Extract the segments that have energy greater than 10% of maximum.
    Calculate the energy in frames that have 25ms frame length and 10ms frame step.
    
    @params:
    waveform (np.ndarray(N)) - the waveform
    Fs (scalar) - sampling rate
    
    @returns:
    segments (list of arrays) - list of the waveform segments where energy is 
       greater than 10% of maximum energy
    '''
    frame_length = int(0.025 * Fs)
    step = int(0.01 * Fs)
    
    if len(waveform) < frame_length:
        return []

    # Calculate number of frames and their respective energies
    num_frames = (len(waveform) - frame_length) // step + 1
    energies = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * step
        end = start + frame_length
        energies[i] = np.sum(waveform[start:end] ** 2)

    max_energy = np.max(energies)
    if max_energy == 0:
        return []

    # Determine which frames are active
    active = energies > 0.1 * max_energy

    # Group contiguous active frames into waveform segments
    segments = []
    in_segment = False
    start_sample = 0

    for i in range(num_frames):
        if active[i]:
            if not in_segment:
                start_sample = i * step
                in_segment = True
        else:
            if in_segment:
                end_sample = (i - 1) * step + frame_length
                segments.append(waveform[start_sample:end_sample])
                in_segment = False

    # If the waveform ends while still in an active segment
    if in_segment:
        end_sample = (num_frames - 1) * step + frame_length
        segments.append(waveform[start_sample:end_sample])

    return segments

def segments_to_models(segments, Fs):
    '''
    Create a model spectrum from each segment:
    Pre-emphasize each segment, then calculate its spectrogram with 4ms frame length and 2ms step,
    then keep only the low-frequency half of each spectrum, then average the low-frequency spectra
    to make the model.
    
    @params:
    segments (list of arrays) - waveform segments that contain speech
    Fs (scalar) - sampling rate
    
    @returns:
    models (list of arrays) - average log spectra of pre-emphasized waveform segments
    '''
    frame_length = int(0.004 * Fs)
    step = int(0.002 * Fs)
    alpha = 0.95  # Standard pre-emphasis coefficient
    models = []
    
    for segment in segments:
        if len(segment) < frame_length:
            continue
            
        # 1. Pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
        emphasized = np.append(segment[0], segment[1:] - alpha * segment[:-1])
        
        # 2. Framing & Spectrogram computation
        num_frames = (len(emphasized) - frame_length) // step + 1
        segment_spectra = []
        
        for i in range(num_frames):
            start = i * step
            end = start + frame_length
            frame = emphasized[start:end]
            
            # Compute Magnitude Spectrum
            fft_vals = np.fft.fft(frame)
            mag = np.abs(fft_vals)
            
            # Compute Log Spectrum (with epsilon to avoid log(0))
            log_spec = np.log(mag + 1e-10)
            
            # 3. Keep only the low-frequency half
            low_freq = log_spec[:frame_length // 2]
            segment_spectra.append(low_freq)
            
        # 4. Average the log spectra across all frames to create the model
        if segment_spectra:
            models.append(np.mean(segment_spectra, axis=0))
            
    return models

def recognize_speech(testspeech, Fs, models, labels):
    '''
    Chop the testspeech into segments using VAD, convert it to models using segments_to_models,
    then compare each test segment to each model using cosine similarity,
    and output the label of the most similar model to each test segment.
    
    @params:
    testspeech (array) - test waveform
    Fs (scalar) - sampling rate
    models (list of Y arrays) - list of model spectra
    labels (list of Y strings) - one label for each model
    
    @returns:
    sims (Y-by-K array) - cosine similarity of each model to each test segment
    test_outputs (list of strings) - recognized label of each test segment
    '''
    # 1. Chop the test speech using Voice Activity Detection
    test_segments = VAD(testspeech, Fs)
    
    # 2. Extract acoustic model spectra for the test segments
    test_models = segments_to_models(test_segments, Fs)
    
    Y = len(models)
    K = len(test_models)
    sims = np.zeros((Y, K))
    
    # 3. Calculate cosine similarity between each template model and test segment model
    for y in range(Y):
        u = models[y]
        norm_u = np.linalg.norm(u)
        for k in range(K):
            v = test_models[k]
            norm_v = np.linalg.norm(v)
            if norm_u > 0 and norm_v > 0:
                sims[y, k] = np.dot(u, v) / (norm_u * norm_v)
            else:
                sims[y, k] = 0.0
                
    # 4. Find the label corresponding to the maximum similarity score
    test_outputs = []
    for k in range(K):
        best_y = np.argmax(sims[:, k])
        test_outputs.append(labels[best_y])
        
    return sims, test_outputs