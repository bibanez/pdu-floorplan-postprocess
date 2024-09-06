import numpy as np
import pandas as pd
from rectsolver import RectSolver
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools


def read_shape_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


def extract_regions(shape):
    unique_regions = np.unique(shape)
    unique_regions = unique_regions[unique_regions >= 0]

    regions = []
    region_positions = []

    for region in unique_regions:
        indices = np.argwhere(shape == region)
        min_row, min_col = indices.min(axis=0)
        max_row, max_col = indices.max(axis=0)

        subarray = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1), dtype=int)
        for r, c in indices:
            subarray[r - min_row, c - min_col] = 1

        regions.append(subarray)
        # Store the top-left position of the region
        region_positions.append((min_row, min_col))

    return regions, region_positions


def process_region(args):
    region, k_rect, region_id = args
    r = RectSolver(region, k_rect)
    results = []
    for rect in r.solve():
        results.append(rect)
    return region_id, region, results


def plot_all_regions_with_rectangles(shape, regions, region_positions, all_rectangles):
    fig, ax = plt.subplots()

    # Display the original shape
    ax.imshow(shape, cmap='tab20', origin='upper')

    # Generate distinct colors for each region
    colors = itertools.cycle(plt.cm.get_cmap('tab10').colors)

    # Overlay rectangles for each region
    for region_id, (rectangles, (min_row, min_col)) in enumerate(zip(all_rectangles, region_positions)):
        color = next(colors)
        for rect in rectangles:
            indices = np.argwhere(rect == 1)

            if indices.size > 0:
                rect_min_row, rect_min_col = indices.min(axis=0)
                rect_max_row, rect_max_col = indices.max(axis=0)

                # Offset the rectangle position based on the original region's top-left position
                rect_patch = patches.Rectangle(
                    (rect_min_col + min_col - 0.5, rect_min_row + min_row - 0.5),
                    rect_max_col - rect_min_col + 1,
                    rect_max_row - rect_min_row + 1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect_patch)

    ax.set_title("All Regions with Rectangles")
    plt.show()


def main():
    file_path = 'shape.csv'
    shape = read_shape_from_csv(file_path)
    regions, region_positions = extract_regions(shape)

    k_rect = 2
    args = [(region, k_rect, i) for i, region in enumerate(regions)]

    with Pool() as pool:
        results = pool.map(process_region, args)

    all_rectangles = [rectangles for _, _, rectangles in results]

    plot_all_regions_with_rectangles(
        shape, regions, region_positions, all_rectangles)


if __name__ == "__main__":
    main()
