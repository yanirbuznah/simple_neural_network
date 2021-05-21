import sys

import numpy as np
import cv2

from main import csv_to_data

def main():
    if len(sys.argv) != 2:
        print("Not enough arguments")
        return

    csv = sys.argv[1]
    print(f"Reading data from: {csv}")
    data, results = csv_to_data(csv)
    for i in range(len(data)):
        bmp_data = data[i] * 255
        bmp_data = bmp_data.astype(np.uint8)
        if results is not None and len(results) > 0:
            classification = results[i].argmax() + 1
        else:
            classification = "all"

        image = bmp_data.reshape(3, 32, 32).transpose(1, 2, 0)
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"/tmp/test/{classification}/test_{i + 1}.bmp", image_cv)

    print()


main()