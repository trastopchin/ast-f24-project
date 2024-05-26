import json
import numexpr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

EPSILON = 1e-4


def read_results(filepath: str) -> list[dict]:
    """Read each line from a results file and deserialize the json strings into a list of dicts."""
    results = []
    with open(filepath, "r") as file:
        for line in file:
            results.append(json.loads(line))
    return results


def filter_invalid(results: list[dict]) -> list[dict]:
    """Remove invalid entries produced before optimal_tolerance == optimal_tolerance was added to POSITIVE_STATUSES."""
    filtered = [
        result
        for result in results
        if not (
            result["type"] == "metamorphic"
            and result["relation_str"] == "optimal_tolerance == optimal_tolerance"
        )
    ]
    return filtered


def filter_optimal_solution(results: list[dict]) -> list[dict]:
    filtered = []

    # Get all consistency results with optimal solutions
    filtered += [
        result
        for result in results
        if result["type"] == "consistency"
        and result["status_gurobi"] == "optimal"
        and result["status_cplex"] == "optimal_tolerance"
    ]

    # Get all metamorphic results with optimal solutions
    filtered += [
        result
        for result in results
        if result["type"] == "metamorphic"
        and (
            result["relation_str"].endswith("optimal == optimal")
            or result["relation_str"].endswith("optimal_tolerance == optimal_tolerance")
        )
    ]

    return filtered


def create_data(results: list[dict]) -> dict:
    types = []
    bugs = []
    errors = []
    files = []
    solvers = []

    # Process consistency results
    for result in results:
        if result["type"] == "consistency":

            # Determine if it's a bug
            obj_val_gurobi = result["obj_val_gurobi"]
            obj_val_cplex = result["obj_val_cplex"]
            error = abs(obj_val_gurobi - obj_val_cplex)
            bug = error > EPSILON

            # Process the result
            types.append(result["type"])
            bugs.append(bug)
            errors.append(error)
            files.append(result["file"])
            solvers.append("both")

    # Process metamorphic results
    for result in results:
        if result["type"] == "metamorphic":

            # Determine if it's a bug
            relation_str = result["relation_str"]
            equation_str = relation_str.split(",")[0]
            lhs_str, rhs_str = equation_str.split(" == ")
            lhs = numexpr.evaluate(lhs_str)
            rhs = numexpr.evaluate(rhs_str)
            error = abs(lhs - rhs)
            bug = error > EPSILON

            # Process the result
            types.append(result["type"])
            bugs.append(bug)
            errors.append(error)
            files.append(result["file"])
            solvers.append(result["solver"])

    # Return data
    data = {
        "type": types,
        "bug": bugs,
        "error": errors,
        "file": files,
        "solver": solvers,
    }
    return data


if __name__ == "__main__":

    results = read_results("./results/results_metamorphic_cplex.txt")
    results += read_results("./results/results.txt")
    results = filter_invalid(results)
    results = filter_optimal_solution(results)
    data = create_data(results)

    # Con
    df = pd.DataFrame(data)
    pivot_table = df.pivot_table(
        index="bug", columns="type", aggfunc="size", fill_value=0
    )

    ax = pivot_table.plot(kind="bar", stacked=True, edgecolor="black")
    ax.set_title("Distribution of Optimal Solution Results by Bug (and Type)")
    ax.set_xlabel("Optimal Solution Bug")
    ax.set_ylabel("Count")
    ax.set_xticks(range(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.index, rotation=0)
    ax.legend(title="Type")

    plt.savefig(
        "distribution_of_optimal_solution_results_by_bug_and_type.pdf", format="pdf"
    )
    # plt.show()
    plt.clf()

    # Extract error values, types, and indices
    # Filter data where bug is true
    bug_true_df = df[df["bug"] == True]
    errors = bug_true_df["error"]
    print(len(errors))

    # Create a log scale histogram
    plt.hist(
        errors, bins=[10**i for i in range(-4, 10)], color="skyblue", edgecolor="black"
    )
    plt.xscale("log")  # Set x-axis to log scale
    plt.xticks([10**i for i in range(-4, 10)])
    plt.title("Histogram of Optimal Solution Bug Errors")
    plt.xlabel("Optimal Solution Bug Error (log scale)")
    plt.ylabel("Frequency")
    plt.savefig("histogram_of_optimal_solution_bug_errors.pdf", format="pdf")
    # plt.show()
