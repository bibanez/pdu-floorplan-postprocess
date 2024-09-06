import numpy as np
import pandas as pd
from rectsolver import RectSolver
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def read_shape_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


def extract_regions(shape):
    unique_regions = np.unique(shape)
    unique_regions = unique_regions[unique_regions >= 0]

    regions = []
    for region in unique_regions:
        indices = np.argwhere(shape == region)
        min_row, min_col = indices.min(axis=0)
        max_row, max_col = indices.max(axis=0)

        subarray = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1), dtype=int)
        for r, c in indices:
            subarray[r - min_row, c - min_col] = 1

        regions.append(subarray)
    return regions


def process_region(args):
    region, k_rect, region_id = args
    r = RectSolver(region, k_rect)
    results = []
    for rect in r.solve():
        results.append(rect)
    return region_id, region, results


def plot_region_with_rectangles(region, rectangles, region_id):
    fig, ax = plt.subplots()

    # Display the region
    ax.imshow(region, cmap='binary', origin='upper')

    # Add rectangles to the plot
    for rect in rectangles:
        # Find the indices where rect equals 1
        indices = np.argwhere(rect == 1)

        if indices.size > 0:
            min_row, min_col = indices.min(axis=0)
            max_row, max_col = indices.max(axis=0)

            # Create a rectangle patch. We subtract 0.5 from the coordinates to
            # align the rectangle with grid lines.
            rect_patch = patches.Rectangle(
                (min_col - 0.5, min_row - 0.5),
                max_col - min_col + 1,
                max_row - min_row + 1,
                linewidth=2,
                edgecolor='red',
                facecolor='none'
            )
            ax.add_patch(rect_patch)

    # Setting title and showing the plot
    ax.set_title(f"Region {region_id + 1} with Rectangles")
    plt.show()


def main():
    file_path = 'shape.csv'
    shape = read_shape_from_csv(file_path)
    regions = extract_regions(shape)

    k_rect = 2
    args = [(region, k_rect, i) for i, region in enumerate(regions)]

    with Pool() as pool:
        results = pool.map(process_region, args)

    for region_id, region, rectangles in results:
        plot_region_with_rectangles(region, rectangles, region_id)


if __name__ == "__main__":
    main()
