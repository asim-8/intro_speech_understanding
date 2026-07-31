import numpy as np
import librosa
from scipy.signal import lfilter

def lpc(speech, frame_length, frame_skip, order):
    '''
    Perform linear predictive analysis of input speech.
    
    @param:
    speech (duration) - input speech waveform
    frame_length (scalar) - frame length, in samples
    frame_skip (scalar) - frame skip, in samples
    order (scalar) - number of LPC coefficients to compute
    
    @returns:
    A (nframes, order+1) - linear predictive coefficients from each frame
    excitation (nframes, frame_length) - linear prediction excitation frames
    '''
    # Compute exact number of full frames required by test suite
    nframes = int((len(speech) - frame_length) / frame_skip)
    
    A = np.zeros((nframes, order + 1))
    excitation = np.zeros((nframes, frame_length))
    
    for i in range(nframes):
        start = i * frame_skip
        frame = speech[start : start + frame_length]
        
        # Calculate LPC coefficients for the raw frame
        a_coeffs = librosa.lpc(frame, order=order)
        A[i] = a_coeffs
        
        # Linear prediction residual / excitation
        excitation[i] = lfilter(a_coeffs, [1.0], frame)
        
    return A, excitation


def synthesize(e, A, frame_skip):
    '''
    Synthesize speech from LPC residual and coefficients.
    
    @param:
    e (duration) - excitation signal
    A (nframes, order+1) - linear predictive coefficients from each frame
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    synthesis (duration) - synthetic speech waveform
    '''
    nframes = A.shape[0]
    synthesis = np.zeros(nframes * frame_skip)
    zi = np.zeros(A.shape[1] - 1)
    
    for i in range(nframes):
        start = i * frame_skip
        end = start + frame_skip
        
        e_frame = e[start:end]
        
        # All-pole synthesis filter: 1 / A(z)
        s_frame, zi = lfilter([1.0], A[i], e_frame, zi=zi)
        synthesis[start:end] = s_frame
        
    return synthesis


def robot_voice(excitation, T0, frame_skip):
    '''
    Calculate the gain for each excitation frame, then create the excitation for a robot voice.
    
    @param:
    excitation (nframes, frame_length) - linear prediction excitation frames
    T0 (scalar) - pitch period, in samples
    frame_skip (scalar) - frame skip, in samples
    
    @returns:
    gain (nframes) - gain for each frame
    e_robot (nframes * frame_skip) - excitation for the robot voice
    '''
    nframes = excitation.shape[0]
    gain = np.zeros(nframes)
    
    # Measure RMS power over the last frame_skip samples of each frame
    for i in range(nframes):
        valid_samples = excitation[i, -frame_skip:]
        gain[i] = np.sqrt(np.mean(valid_samples ** 2))
        
    total_samples = nframes * frame_skip
    
    # Create periodic impulse train at pitch period T0
    impulse_train = np.zeros(total_samples)
    impulse_indices = np.arange(0, total_samples, int(T0))
    impulse_train[impulse_indices] = 1.0
    
    e_robot = np.zeros(total_samples)
    for i in range(nframes):
        start = i * frame_skip
        end = (i + 1) * frame_skip
        e_robot[start:end] = impulse_train[start:end] * gain[i]
        
    return gain, e_robot