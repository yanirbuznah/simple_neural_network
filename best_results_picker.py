import glob
import os
from pprint import pprint

import pandas as pd

TOP_N = 10
DIR = "."

BEST_RESULTS = {}

for csv_file in glob.glob(os.path.join(DIR, "**", "results.csv"), recursive=False):
    base_path = os.path.dirname(csv_file)
    results_csv = pd.read_csv(csv_file)
    validate_results = results_csv["validate_accuracy"]
    BEST_RESULTS[csv_file] = max(validate_results)

sorted_results = list((k, v) for k, v in sorted(BEST_RESULTS.items(), reverse=True, key=lambda item: item[1]))
pprint(sorted_results[:TOP_N])

