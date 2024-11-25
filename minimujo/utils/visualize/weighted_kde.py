import warnings

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

class WeightedKDEHeatmap:

    def __init__(self, xy_range=(0,0,1,1), heatmap_resolution=(200,200), kernel_size=9, decay=1):
        assert kernel_size % 2 == 1, f"kernel_size {kernel_size} should be odd"

        self.kernel_size = kernel_size
        self.sigma = kernel_size / 4
        self.kernel = WeightedKDEHeatmap.create_gaussian_kernel(self.kernel_size, self.sigma)
        self.decay = decay

        self.heatmap_shape = heatmap_resolution[::-1]
        self.heatmap = np.full(heatmap_resolution, np.nan)
        self.densitymap = np.full(heatmap_resolution, np.nan)
        self.norm_factor = 0

        self.half_size = self.kernel_size // 2
        self.half_vector = np.array([self.half_size, self.half_size])
        self.lower_bounds = np.zeros(2, dtype=int)
        self.upper_bounds = np.array([self.heatmap_shape[0], self.heatmap_shape[1]], dtype=int)
        self.kernel_shape = np.array([kernel_size, kernel_size], dtype=int)
        self.kernel_area = self.kernel.sum()

        # for scaling points
        self.range_corner = np.array(xy_range[:2])
        self.range_scale = self.upper_bounds / np.array(xy_range[2:])

    @staticmethod
    def create_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        return kernel / np.sum(kernel)
    
    def add_batch(self, points: np.ndarray, values: np.ndarray) -> None:
        if points.shape[0] == 0:
            return
        assert points.shape[0] == values.shape[0], f"Invalid shapes: {points.shape}, {values.shape}"
        points = ((points - self.range_corner) * self.range_scale).astype(int)

        if self.decay < 1:
            self.densitymap *= self.decay

        for point, value in zip(points, values):
            lower = (point - self.half_vector).astype(int)
            upper = (point + self.half_vector + 1).astype(int)

            # indices for full image
            x_start, y_start = clipped_lower = np.max([lower, self.lower_bounds], axis=0)
            x_end, y_end = clipped_upper = np.min([upper, self.upper_bounds], axis=0)
                                
            # indices for kernel
            kernel_x_start, kernel_y_start = clipped_lower - lower
            kernel_x_end, kernel_y_end = self.kernel_shape - (clipped_upper - upper)

            clipped_size = np.min(clipped_upper - clipped_lower)
            if clipped_size < self.kernel_size:
                if clipped_size < 0:
                    # fully out of bounds
                    print(f'Warning: failed to visualize {tuple((point / self.range_scale) + self.range_corner)} with value {value}')
                    continue
                
            old_weight = np.nan_to_num(self.densitymap[y_start:y_end, x_start:x_end], copy=False)
            added_weight = self.kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
            new_weight = old_weight + added_weight
            old_value = np.nan_to_num(self.heatmap[y_start:y_end, x_start:x_end], copy=False)
            self.heatmap[y_start:y_end, x_start:x_end] = (old_weight * old_value + added_weight * value) / new_weight
            self.densitymap[y_start:y_end, x_start:x_end] = new_weight

    def add_single(self, point, weight):
        self.add_batch(np.array([point]), np.array([weight]))

    @property
    def densitymap_normalized(self):
        return self.densitymap / np.nansum(self.densitymap)
    
def get_heatmap_figure(heatmap: WeightedKDEHeatmap, extent=(0, 1, 0, 1), is_log_scale=False, origin_corner='lower', axis_labels=('X', 'Y', 'Value'), title='', color_map='plasma', value_min=None, value_max=None, log_density=False):
    norm = LogNorm() if is_log_scale else None
    figure, _ = plt.subplots()
    logmap = heatmap.densitymap_normalized if log_density else heatmap.heatmap
    plt.imshow(logmap, origin=origin_corner, cmap=color_map, extent=extent, norm=norm, vmin=value_min, vmax=value_max)
    plt.colorbar(label=axis_labels[2])
    plt.title(title)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])
    return figure

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # extents = 
    kde_heatmap = WeightedKDEHeatmap(decay=.98)

    # center = np.array([10.,10.])
    # for i in range(800):
    #     center += np.array([0.1, 0.15])
    #     point = (np.random.uniform(-10, 10, 2) + center).astype(int)
    #     weight = np.random.uniform(0,1)
    #     kde_heatmap.add_single(point, 1)

    batch_size = 10
    center = np.array([0.02,0.02])
    mean = 4
    for i in range(80):
        mean += 0.2
        if i < 40:
            center += np.array([0.01, 0.015])
        else:
            center -= np.array([0.01, 0.01])
        points = np.random.uniform(-0.05, 0.05, (batch_size,2)) + center
        weights = np.random.uniform(0,1, batch_size) + mean
        weights = np.ones(batch_size)
        kde_heatmap.add_batch(points, weights)

    # points = np.random.uniform(low=[1, -4], high=[4,-1], size=(10,2))
    # weights = np.ones(10)
    # kde_heatmap.add_batch(points, weights)

    # Plot the heatmap
    plt.imshow(kde_heatmap.densitymap_normalized, origin='lower', cmap='viridis', extent=(0,1,0,1))
    plt.colorbar(label='Density')
    plt.title('Heat Map with Precomputed Gaussian Kernel Stamp')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()