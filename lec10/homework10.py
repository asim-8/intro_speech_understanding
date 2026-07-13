import numpy as np
import torch
import torch.nn as nn

def get_features(waveform, Fs):
    '''
    Get features from a waveform.
    @params:
    waveform (numpy array) - the waveform
    Fs (scalar) - sampling frequency.

    @return:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels (integers)
    '''
    # ------------------ 1. Compute Acoustic Features ------------------
    feat_len = int(0.004 * Fs)
    feat_step = int(0.002 * Fs)
    alpha = 0.95
    
    # Pre-emphasis
    emphasized = np.append(waveform[0], waveform[1:] - alpha * waveform[:-1])
    
    # Framing for features
    num_feat_frames = (len(emphasized) - feat_len) // feat_step + 1
    feature_list = []
    
    for i in range(num_feat_frames):
        start = i * feat_step
        end = start + feat_len
        frame = emphasized[start:end]
        mag_spec = np.abs(np.fft.fft(frame))
        log_spec = np.log(mag_spec + 1e-10)
        feature_list.append(log_spec[:feat_len // 2])
        
    features = np.array(feature_list) # Shape: (NFRAMES, NFEATS)
    NFRAMES = features.shape[0]

    # ------------------ 2. Voice Activity Detection (VAD) ------------------
    vad_len = int(0.025 * Fs)
    vad_step = int(0.01 * Fs)
    
    num_vad_frames = (len(waveform) - vad_len) // vad_step + 1
    energies = np.zeros(num_vad_frames)
    for i in range(num_vad_frames):
        start = i * vad_step
        end = start + vad_len
        energies[i] = np.sum(waveform[start:end] ** 2)
        
    max_energy = np.max(energies) if len(energies) > 0 else 0
    active = energies > 0.1 * max_energy if max_energy > 0 else np.zeros(num_vad_frames, dtype=bool)
    
    # Locate contiguous speech segments from VAD
    segments_bounds = []
    in_segment = False
    start_sample = 0
    
    for i in range(num_vad_frames):
        if active[i]:
            if not in_segment:
                start_sample = i * vad_step
                in_segment = True
        else:
            if in_segment:
                end_sample = (i - 1) * vad_step + vad_len
                segments_bounds.append((start_sample, end_sample))
                in_segment = False
    if in_segment:
        segments_bounds.append((start_sample, (num_vad_frames - 1) * vad_step + vad_len))
        
    # ------------------ 3. Map VAD segments to Labels ------------------
    labels = np.zeros(NFRAMES, dtype=int)
    
    # Map each feature frame index back to its physical sample timeline
    for i in range(NFRAMES):
        frame_sample_center = i * feat_step + (feat_len // 2)
        
        # Check which VAD speech segment this feature frame falls into
        for idx, (start, end) in enumerate(segments_bounds):
            if start <= frame_sample_center < end:
                # Segment IDs start at 1 (0 is background/silent)
                labels[i] = idx + 1
                break

    return features, labels

def train_neuralnet(features, labels, iterations):
    '''
    @param:
    features (NFRAMES,NFEATS) - numpy array of feature vectors
    labels (NFRAMES) - numpy array of labels (integers)
    iterations (scalar) - number of iterations of training

    @return:
    model - a neural net model created in pytorch, and trained using the provided data
    lossvalues (numpy array, length=iterations) - the loss value achieved on each iteration of training
    '''
    NFEATS = features.shape[1]
    num_classes = int(1 + np.max(labels))
    
    # Construct sequential model network layer structure
    model = nn.Sequential(
        nn.LayerNorm(NFEATS),
        nn.Linear(NFEATS, num_classes)
    )
    
    # Setup optimization parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Convert numpy data to PyTorch format tensors
    X = torch.from_numpy(features).float()
    y = torch.from_numpy(labels).long()
    
    lossvalues = np.zeros(iterations)
    
    # Training Loop
    for it in range(iterations):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X)
        loss = criterion(outputs, y)
        
        loss.backward()
        optimizer.step()
        
        lossvalues[it] = loss.item()
        
    return model, lossvalues

def test_neuralnet(model, features):
    '''
    @param:
    model - a neural net model created in pytorch, and trained
    features (NFRAMES, NFEATS) - numpy array
    
    @return:
    probabilities (NFRAMES, NLABELS) - model output, transformed by softmax, detach().numpy().
    '''
    model.eval()
    X = torch.from_numpy(features).float()
    
    with torch.no_grad():
        logits = model(X)
        # Apply Softmax activation mapping along classes dimension
        probs = torch.softmax(logits, dim=-1)
        
    return probs.detach().numpy()