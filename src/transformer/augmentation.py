import numpy as np

class DataAugmenter:
    def __init__(self, noise_std=0.005, scale_range=(0.95, 1.05), mask_prob=0.05):
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.mask_prob = mask_prob

    def add_noise(self, x):
        """Add Gaussian noise to features."""
        noise = np.random.normal(0, self.noise_std, x.shape)
        return x + noise

    def scale_features(self, x):
        """Scale features by a random factor."""
        scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
        return x * scale

    def mask_features(self, x):
        """Randomly mask out some time steps (dropout on input)."""
        mask = np.random.binomial(1, 1 - self.mask_prob, x.shape[0])
        # Expand mask to match feature dim
        mask = mask[:, np.newaxis]
        return x * mask

    def augment(self, x):
        """Apply random augmentations."""
        # Randomly choose one or more augmentations
        if np.random.rand() > 0.5:
            x = self.add_noise(x)
        if np.random.rand() > 0.5:
            x = self.scale_features(x)
        if np.random.rand() > 0.7: # Increased frequency for regularization
            x = self.mask_features(x)
        return x
