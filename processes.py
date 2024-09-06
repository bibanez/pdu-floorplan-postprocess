import numpy as np
import pandas as pd
from rectsolver import RectSolver
import multiprocessing as mp


def read_shape_from_csv(file_path):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path, header=None)
    return df.to_numpy()


def extract_regions(shape):
    # Get unique region indices excluding 0
    unique_regions = np.unique(shape)
    unique_regions = unique_regions[unique_regions != 0]

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

        regions.append((region, subarray))

    return regions


def solve_region(region_data, conn):
    region, subarray = region_data
    k_rect = 3  # Adjust this as needed
    r = RectSolver(subarray, k_rect)
    results = [rect for rect in r.solve()]
    conn.send((region, results))
    conn.close()


def main():
    # Read shape from CSV
    file_path = 'shape.csv'  # Update this with your CSV file path
    shape = read_shape_from_csv(file_path)

    # Extract regions from the shape
    regions = extract_regions(shape)

    processes = []
    parent_conns = []

    # Create a process for each region
    for region_data in regions:
        parent_conn, child_conn = mp.Pipe()
        p = mp.Process(target=solve_region, args=(region_data, child_conn))
        processes.append(p)
        parent_conns.append(parent_conn)
        p.start()

    # Collect results from all processes
    for parent_conn in parent_conns:
        region, rects = parent_conn.recv()
        print(f"Results for region {region}:")
        for rect in rects:
            print(rect)
            print("\n" + "="*20 + "\n")

    # Ensure all processes have finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
