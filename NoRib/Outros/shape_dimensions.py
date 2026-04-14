"""
python shape_dimensions.py partition_data_teardrop.mat partition_data_square.mat
"""
import scipy.io
import numpy as np
import glob
import sys

if len(sys.argv) > 1:
    files = sys.argv[1:]
else:
    files = sorted(glob.glob('partition_data_*.mat'))
if not files:
    print("No partition_data_*.mat files found in current directory.")
    exit()

print(f"{'Shape':<20} {'X range':>12} {'Y range':>12} {'Width':>10} {'Height':>10}")
print("-" * 68)

for f in files:
    data = scipy.io.loadmat(f)
    nodes = data['nodesCoords']
    x_min, x_max = nodes[:, 0].min(), nodes[:, 0].max()
    y_min, y_max = nodes[:, 1].min(), nodes[:, 1].max()
    width = x_max - x_min
    height = y_max - y_min
    name = f.replace('partition_data_', '').replace('.mat', '')
    print(f"{name:<20} [{x_min:+.1f}, {x_max:+.1f}] [{y_min:+.1f}, {y_max:+.1f}] {width:>10.1f} {height:>10.1f}")
