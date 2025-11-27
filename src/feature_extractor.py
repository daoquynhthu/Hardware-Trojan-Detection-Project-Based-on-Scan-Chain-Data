import numpy as np
from scipy.stats import entropy, pearsonr
from scipy.fft import fft

def extract_features_from_design(seqs):
    """
    Extract features from register sequences.
    
    Args:
        seqs: np.ndarray of shape [N_cycles, N_registers], containing 0/1 binary values.
        
    Returns:
        features: np.ndarray of shape [N_registers, N_features]
    """
    n_cycles, n_regs = seqs.shape
    
    # --- Pre-computation for vectorized operations ---
    
    # 1. Global Statistics
    means = np.mean(seqs, axis=0)
    variances = np.var(seqs, axis=0)
    
    # Toggle rate
    diffs = np.diff(seqs, axis=0) # shape [N_cycles-1, N_regs]
    abs_diffs = np.abs(diffs)
    toggle_rates = np.mean(abs_diffs, axis=0)
    
    # Entropy (Binary entropy based on probability of 1)
    # H(p) = -p log2 p - (1-p) log2 (1-p)
    p = np.clip(means, 1e-9, 1 - 1e-9)
    entropies = -p * np.log2(p) - (1-p) * np.log2(1-p)
    
    # Static Ratio: max(mean, 1-mean)
    static_ratios = np.maximum(means, 1-means)
    
    # Duty Cycle (same as mean)
    duty_cycles = means
    
    # For Neighbor Correlation Optimization
    # Normalize sequences: Z = (X - mean) / std
    # Avoid division by zero for constant registers
    stds = np.sqrt(variances)
    valid_std_mask = stds > 1e-9
    
    # Z_seqs shape: [N_cycles, N_regs]
    Z_seqs = np.zeros_like(seqs, dtype=float)
    Z_seqs[:, valid_std_mask] = (seqs[:, valid_std_mask] - means[valid_std_mask]) / stds[valid_std_mask]
    
    final_features = []
    
    for i in range(n_regs):
        reg_feats = []
        sig = seqs[:, i]
        
        # --- 4.1 Global Features ---
        reg_feats.append(means[i])          # mean
        reg_feats.append(variances[i])      # variance
        reg_feats.append(toggle_rates[i])   # toggle_rate
        reg_feats.append(entropies[i])      # entropy
        
        # Longest Run 0/1, Num Runs
        # Find runs
        d = np.diff(sig)
        change_indices = np.where(d != 0)[0]
        
        if len(change_indices) == 0:
            # Constant signal
            max_run_0 = n_cycles if sig[0] == 0 else 0
            max_run_1 = n_cycles if sig[0] == 1 else 0
            num_runs = 1
            switching_peaks = 0
        else:
            # Boundaries for runs
            boundaries = np.concatenate(([-1], change_indices, [n_cycles-1]))
            run_lengths = np.diff(boundaries)
            
            # Values of runs (check value at start of each run)
            # Run starts at index: boundaries[:-1] + 1
            run_starts = boundaries[:-1] + 1
            run_values = sig[run_starts]
            
            max_run_0 = np.max(run_lengths[run_values == 0]) if np.any(run_values == 0) else 0
            max_run_1 = np.max(run_lengths[run_values == 1]) if np.any(run_values == 1) else 0
            num_runs = len(run_lengths)
            # Fix: Ensure d is treated as array for sum
            switching_peaks = np.sum(np.abs(np.array(d))) # Total transitions

        reg_feats.append(max_run_0)         # longest_run_0
        reg_feats.append(max_run_1)         # longest_run_1
        reg_feats.append(num_runs)          # num_runs
        reg_feats.append(static_ratios[i])  # static_ratio
        reg_feats.append(switching_peaks)   # switching_peaks
        reg_feats.append(duty_cycles[i])    # duty_cycle
        
        # --- 4.2 Window Features ---
        # Windows: 8, 32, 128
        for w in [8, 32, 128]:
            if n_cycles < w:
                # Fallback if sequence is too short
                reg_feats.extend([means[i], 0, toggle_rates[i], 0])
                continue
            
            # Use convolution for sliding mean
            kernel = np.ones(w) / w
            
            # Sliding Mean
            w_means = np.convolve(sig, kernel, mode='valid')
            reg_feats.append(np.mean(w_means)) # mean(mean)
            reg_feats.append(np.std(w_means))  # std(mean)
            
            # Sliding Toggle Rate
            # Abs diff has length N-1
            sig_abs_diff = abs_diffs[:, i]
            if len(sig_abs_diff) >= w:
                w_toggles = np.convolve(sig_abs_diff, kernel, mode='valid')
                reg_feats.append(np.mean(w_toggles)) # mean(toggle)
                reg_feats.append(np.std(w_toggles))  # std(toggle)
            else:
                reg_feats.append(toggle_rates[i])
                reg_feats.append(0)
                
        # --- 4.3 Markov Features ---
        # P(0->0), P(0->1), P(1->0), P(1->1)
        pairs = sig[:-1] * 2 + sig[1:] # Map pairs to 0,1,2,3
        counts = np.bincount(pairs.astype(int), minlength=4)
        total_trans = len(pairs)
        if total_trans > 0:
            probs = counts / total_trans
        else:
            probs = [0, 0, 0, 0]
        reg_feats.extend(probs)
        
        # --- 4.4 FFT Features ---
        # Top 5 non-zero freq magnitudes (skipping DC)
        if n_cycles > 1:
            fft_vals = np.abs(np.array(fft(sig)))
            fft_vals = fft_vals[1:] # Skip DC
            # Take first 5
            if len(fft_vals) >= 5:
                reg_feats.extend(fft_vals[:5])
            else:
                reg_feats.extend(fft_vals)
                reg_feats.extend([0] * (5 - len(fft_vals)))
        else:
            reg_feats.extend([0] * 5)
            
        # --- 4.5 Neighbor Features ---
        # Correlation with i-2, i-1, i+1, i+2
        # Use pre-computed Z_seqs for speed: Corr = Z_i . Z_j / N
        # Note: Pearson corr uses N-1 in denominator for sample std, or N?
        # np.corrcoef uses N-1. Our Z uses N (if we used np.std).
        # np.std is population std by default? No, it's population. ddof=0.
        # Let's stick to dot product logic. Corr ~ dot(Z_i, Z_j) / n_cycles
        
        z_i = Z_seqs[:, i]
        
        for offset in [-2, -1, 1, 2]:
            neighbor_idx = i + offset
            if 0 <= neighbor_idx < n_regs:
                if not valid_std_mask[i] or not valid_std_mask[neighbor_idx]:
                    corr = 0
                else:
                    z_j = Z_seqs[:, neighbor_idx]
                    corr = np.dot(z_i, z_j) / n_cycles
                reg_feats.append(corr)
            else:
                reg_feats.append(0)
                
        final_features.append(reg_feats)
        
    return np.array(final_features)
