# Attention Visualization Guide

This folder contains visualization plots showing how the Transformer model locates Trojans.

## How to Read the Plots

Each image (e.g., `vis_designX_stepY.png`) contains two subplots:

1.  **Top Graph (Red Line)**: Ground Truth Trojan Label
    *   **Value 1 (Spike)**: Indicates the exact time step where the Hardware Trojan is active/triggered.
    *   **Value 0 (Flat)**: Normal behavior.

2.  **Bottom Graph (Blue Area)**: Model Attention Weights
    *   This shows "where the model is looking" to make its decision.
    *   **High Peaks**: The model considers these time steps critical for classifying the sequence as "Trojan".
    *   **Low/Flat**: The model ignores these time steps.

## Interpretation

*   **Ideal Case**: The Blue Peak aligns perfectly with the Red Spike. This means the model has successfully learned to "attend" to the specific Trojan trigger pattern without being explicitly told where it is (Weakly Supervised Learning).
*   **Offset**: If the Blue Peak is slightly shifted (e.g., 5-10 steps before/after), it's normal. The model might be detecting the "setup" or the "consequence" of the Trojan rather than the trigger itself.
*   **Noise**: If the Blue Area is messy or uniform, the model might be relying on global statistics (like average power consumption) rather than specific trigger patterns.
