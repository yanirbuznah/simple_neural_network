import sys

import numpy as np
from PIL import Image

from main import csv_to_train_data

def main():
    if len(sys.argv) != 2:
        print("Not enough arguments")
        return

    csv = sys.argv[1]
    print(f"Reading data from: {csv}")
    data, _ = csv_to_train_data(csv)
    for i in range(len(data)):
        bmp_data = data[i] * 255
        bmp_data = bmp_data.astype(np.uint8)
        Image.fromarray(bmp_data.reshape((96, 32))).save(f"/tmp/test/test{i}.bmp")
    print()


main()