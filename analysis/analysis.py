import json
from pathlib import Path
import numexpr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-4

def get_n_input_files(input_dir: str) -> int:
    return len([file for file in Path(input_dir).iterdir() if file.is_file()])

def read_results(filepath: str) -> list[dict]:
    """Read each line from a results file and deserialize the json strings into a list of dicts."""
    results = []
    with open(filepath, "r") as file:
        for line in file:
            results.append(json.loads(line))
    return results

def filter_missing(results: list[dict], missing_results: list[dict]) -> list[dict]:
    """Remove {'type': 'metamorphic', 'solver': 'cplex'} results for each missing result (produced before "optimal_tolerance" was added to POSITIVE_STATUSES)."""
    missing_result_files = [missing_result["file"] for missing_result in missing_results]
    filtered = [
        result for result in results if
        not (result["type"] == "metamorphic" and
             result["file"] in missing_result_files and
             result["solver"] == "cplex")
    ]
    return filtered


def create_data(results: list[dict]) -> dict:
    types = []
    bugs = []
    errors = []
    files = []
    solvers = []
    
    for result in results:
        if result["type"] == "consistency":
            pass
    
    # Return data
    data = {
        "type" : types,
        "bug" : bugs,
        "error" : errors,
        "file" : files,
        "solver" : solvers
    }
    return data

if __name__ == "__main__":
    
    # Get the total number of input files
    n_input_files = get_n_input_files("../input")
    print(f"Number of input files: {n_input_files}")
    
    # Read and combine the results and missing results
    results = read_results("../results/results.txt")
    results_missing = read_results("../results/results_missing.txt")
    results = filter_missing(results, results_missing)
    results += results_missing
    
    # Result statistics
    print(f"Total number of collected results: {len(results)}")
    processed_files = [result["file"] for result in results]
    n_processed_files = len(set(processed_files))
    print(f"Total number of processed files: {n_processed_files}")
    print(f"Percentage of processed files: {100 * n_processed_files / n_input_files:.2f}%")
    # Sanity check: each file produces 1 consistency check and 2 metamorphic checks * 2 solvers = 5 results
    assert n_processed_files * 5 == len(results)
    
    # Create the data
    data = create_data(results)
    df = pd.DataFrame(data)