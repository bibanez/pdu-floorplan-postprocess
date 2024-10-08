import numpy as np
import pandas as pd
from rectsolver import RectSolver


def read_shape_from_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


def extract_regions(shape):
    # Get unique region indices
    unique_regions = np.unique(shape)
    unique_regions = unique_regions[unique_regions >= 0]

    regions = []

    for region in unique_regions:
        # Find the bounding box of the region
        indices = np.argwhere(shape == region)
        min_row, min_col = indices.min(axis=0)
        max_row, max_col = indices.max(axis=0)

        # Create a subarray for the region with 1 and 0
        subarray = np.zeros(
            (max_row - min_row + 1, max_col - min_col + 1), dtype=int)
        for r, c in indices:
            subarray[r - min_row, c - min_col] = 1

        regions.append(subarray)

    return regions


def main():
    # Read shape from CSV
    file_path = 'shape.csv'  # Update this with your CSV file path
    shape = read_shape_from_csv(file_path)

    # Extract regions from the shape
    regions = extract_regions(shape)

    # Use RectSolver for each region
    k_rect = 3  # You may want to adjust this for each region
    for i, region in enumerate(regions):
        print(f"Processing region {i+1}:")
        print(region)
        r = RectSolver(region, k_rect)
        for rect in r.solve():
            print(rect)
            print("\n" + "="*20 + "\n")


if __name__ == "__main__":
    main()
