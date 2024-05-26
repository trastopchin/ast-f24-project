import json
from pathlib import Path
import numexpr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Statuses
OPTIMAL_STATUSES = ["optimal", "optimal_tolerance"]
TIME_LIMIT_FEASIBLE_STATUSES = ["time_limit_feasible"]
TIME_LIMIT_INFEASIBLE_STATUSES = ["time_limit_infeasible"]
INFEASIBLE_STATUSES = ["infeasible"]

# Error computable statuses
ERROR_COMPUTABLE_STATUSES = OPTIMAL_STATUSES + TIME_LIMIT_FEASIBLE_STATUSES

# Error epsilon
EPSILON = 1e-4

def get_n_input_files(input_dir: str) -> int:
    """Determine the number of input files."""
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

def statuses_are_error_computable(status1: str, status2: str) -> bool:
    """Determine if we can compute the error based on the statuses."""
    return status1 in ERROR_COMPUTABLE_STATUSES and status2 in ERROR_COMPUTABLE_STATUSES
    

def statuses_and_value_match_to_bug(status1: str, status2: str, value_match: Optional[bool]) -> str:
    """Compute the bug based on the statuses and value_match."""
    
    # First row of table 2
    if status1 in OPTIMAL_STATUSES and status2 in OPTIMAL_STATUSES:
        return "no" if value_match else "correctness"
    elif status1 in OPTIMAL_STATUSES and status2 in TIME_LIMIT_FEASIBLE_STATUSES:
        return "determination" if value_match else "efficiency"
    elif status1 in OPTIMAL_STATUSES and status2 in TIME_LIMIT_INFEASIBLE_STATUSES:
        return "efficiency"
    elif status1 in OPTIMAL_STATUSES and status2 in INFEASIBLE_STATUSES:
        return "correctness"
    # Second row of table 2
    elif status1 in TIME_LIMIT_FEASIBLE_STATUSES and status2 in OPTIMAL_STATUSES:
        return "determination" if value_match else "efficiency"
    elif status1 in TIME_LIMIT_FEASIBLE_STATUSES and status2 in TIME_LIMIT_FEASIBLE_STATUSES:
        return "no" if value_match else "efficiency"
    elif status1 in TIME_LIMIT_FEASIBLE_STATUSES and status2 in TIME_LIMIT_INFEASIBLE_STATUSES:
        return "efficiency"
    elif status1 in TIME_LIMIT_FEASIBLE_STATUSES and status2 in INFEASIBLE_STATUSES:
        return "correctness"
    # Third row of table 2
    elif status1 in TIME_LIMIT_INFEASIBLE_STATUSES and status2 in OPTIMAL_STATUSES:
        return "efficiency"
    elif status1 in TIME_LIMIT_INFEASIBLE_STATUSES and status2 in TIME_LIMIT_FEASIBLE_STATUSES:
        return "efficiency"
    elif status1 in TIME_LIMIT_INFEASIBLE_STATUSES and status2 in TIME_LIMIT_INFEASIBLE_STATUSES:
        return "no"
    elif status1 in TIME_LIMIT_INFEASIBLE_STATUSES and status2 in INFEASIBLE_STATUSES:
        return "determination"
    # Fourth row of table 2
    elif status1 in INFEASIBLE_STATUSES and status2 in OPTIMAL_STATUSES:
        return "correctness"
    elif status1 in INFEASIBLE_STATUSES and status2 in TIME_LIMIT_FEASIBLE_STATUSES:
        return "correctness"
    elif status1 in INFEASIBLE_STATUSES and status2 in TIME_LIMIT_INFEASIBLE_STATUSES:
        return "determination"
    elif status1 in INFEASIBLE_STATUSES and status2 in INFEASIBLE_STATUSES:
        return "no"
    # Unclassified case
    else:
        return "unclassified"
    

def create_data(results: list[dict]) -> dict:
    """Transform the results into data that can be used to create a pandas.DataFrame."""
    types = []
    bugs = []
    errors = []
    files = []
    solvers = []
    status1_list = []
    status2_list = []
    
    for result in results:
        result_type = result["type"]
        result_bug = None
        result_error = None
        result_file = result["file"]
        result_solver = None
        result_status1 = None
        result_status2 = None
        
        # Consistency results
        if result_type == "consistency":
            result_solver = "both"
            
            # Compute the error, if possible
            result_status1 = result["status_gurobi"]
            result_status2 = result["status_cplex"]
            value_match = None
            
            if statuses_are_error_computable(result_status1, result_status2):
                obj_val_gurobi = result["obj_val_gurobi"]
                obj_val_cplex = result["obj_val_cplex"]
                result_error = abs(obj_val_gurobi - obj_val_cplex)
                value_match = result_error <= EPSILON
            
            # Determine the bug type
            result_bug = statuses_and_value_match_to_bug(result_status1, result_status2, value_match)
        
        # Metamorphic results
        elif result_type == "metamorphic":
            result_solver = result["solver"]
            
            # Parse the relation_str to get the statuses
            relation_str = result["relation_str"]
            relation_str_split = relation_str.split(", ")
            status_str = relation_str_split[0] if len(relation_str_split) == 1 else relation_str_split[1]
            result_status1, result_status2 = status_str.split(" == ")
            value_match = None
            
            # Compute the error, if possible
            if statuses_are_error_computable(result_status1, result_status2):
                equation_str = relation_str_split[0]
                lhs_str, rhs_str = equation_str.split(" == ")
                lhs = numexpr.evaluate(lhs_str)
                rhs = numexpr.evaluate(rhs_str)
                result_error = abs(lhs - rhs)
                value_match = result_error <= EPSILON
                
            # Determine the bug type
            result_bug = statuses_and_value_match_to_bug(result_status1, result_status2, value_match)

        # Append the type, bug, error, file, and solver
        types.append(result_type)
        bugs.append(result_bug)
        errors.append(result_error)
        files.append(result_file)
        solvers.append(result_solver)
        status1_list.append(result_status1)
        status2_list.append(result_status2)
        
    
    # Return data
    data = {
        "type" : types,
        "bug" : bugs,
        "error" : errors,
        "file" : files,
        "solver" : solvers,
        "status1" : status1_list,
        "status2" : status2_list
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
