import numpy as np

class WeightedKDEHeatmap:

    def __init__(self, heatmap_size=(200,200), kernel_size=9, channels=1, decay=1):
        assert kernel_size % 2 == 1, f"kernel_size {kernel_size} should be odd"

        self.kernel_size = kernel_size
        self.sigma = kernel_size / 4
        self.kernel = WeightedKDEHeatmap.create_gaussian_kernel(self.kernel_size, self.sigma)
        self.decay = decay

        self.heatmap_shape = heatmap_size
        self.weightmap = np.zeros(heatmap_size)
        self.densitymap = np.zeros(heatmap_size)
        self.norm_factor = 0

        self.half_size = self.kernel_size // 2
        self.half_vector = np.array([self.half_size, self.half_size])
        self.lower_bounds = np.zeros(2, dtype=int)
        self.upper_bounds = np.array(self.heatmap_shape, dtype=int)
        self.kernel_shape = np.array([kernel_size, kernel_size], dtype=int)
        self.kernel_area = self.kernel.sum()

    @staticmethod
    def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)
    
    def add_batch(self, points: np.ndarray, weights: np.ndarray) -> None:
        assert points.shape[0] > 0 and points.shape[0] == weights.shape[0]
        points = (points * np.array(self.heatmap_shape)).astype(int)
        print(points.mean(), np.array(self.heatmap_shape))

        old_norm_factor = self.norm_factor
        self.norm_factor = self.decay * old_norm_factor
        if old_norm_factor == 0:
            old_norm_factor = 1

        for point, weight in zip(points, weights):
            lower = (point - self.half_vector).astype(int)
            upper = (point + self.half_vector + 1).astype(int)

            # indices for full image
            x_start, y_start = clipped_lower = np.max([lower, self.lower_bounds], axis=0)
            x_end, y_end = clipped_upper = np.min([upper, self.upper_bounds], axis=0)
                                
            # indices for kernel
            kernel_x_start, kernel_y_start = clipped_lower - lower
            kernel_x_end, kernel_y_end = self.kernel_shape - (clipped_upper - upper)

            kernel_area = self.kernel_area
            clipped_size = np.min(clipped_upper - clipped_lower)
            if clipped_size < self.kernel_size:
                if clipped_size < 0:
                    # fully out of bounds
                    continue
                # don't unnecessarily compute area
                kernel_area = self.kernel[kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end].sum()
                
            self.norm_factor += kernel_area
            
            density_weight = 1 / (self.decay * old_norm_factor)
            self.densitymap[x_start:x_end, y_start:y_end] += density_weight * self.kernel[kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end]
            self.weightmap[x_start:x_end, y_start:y_end] += weight * density_weight * self.kernel[kernel_x_start:kernel_x_end, kernel_y_start:kernel_y_end]

        # normalize the density
        norm_ratio = old_norm_factor / self.norm_factor
        self.densitymap *= (self.decay * norm_ratio)
        self.weightmap *= (self.decay * norm_ratio)
        
    def add_single(self, point, weight):
        self.add_batch(np.array([point]), np.array([weight]))

    @property
    def heatmap(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(self.densitymap == 0, np.nan, self.weightmap / self.densitymap)

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    kde_heatmap = WeightedKDEHeatmap(decay=0.9)

    # center = np.array([10.,10.])
    # for i in range(800):
    #     center += np.array([0.1, 0.15])
    #     point = (np.random.uniform(-10, 10, 2) + center).astype(int)
    #     weight = np.random.uniform(0,1)
    #     kde_heatmap.add_single(point, 1)

    batch_size = 10
    center = np.array([0.02,0.02])
    mean = 0
    for i in range(80):
        mean += 0.02
        if i < 40:
            center += np.array([0.01, 0.015])
        else:
            center -= np.array([0.01, 0.015])
        points = np.random.uniform(-0.1, 0.1, (batch_size,2)) + center
        weights = np.random.uniform(0,1, batch_size) + mean
        # weights = np.ones(batch_size)
        kde_heatmap.add_batch(points, weights)

    # Plot the heatmap
    plt.imshow(kde_heatmap.heatmap, origin='lower', cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Heat Map with Precomputed Gaussian Kernel Stamp')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()