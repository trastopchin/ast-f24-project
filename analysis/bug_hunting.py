import pandas as pd

# Statuses
OPTIMAL_STATUSES = ["optimal", "optimal_tolerance"]
TIME_LIMIT_FEASIBLE_STATUSES = ["time_limit_feasible"]
TIME_LIMIT_INFEASIBLE_STATUSES = ["time_limit_infeasible"]
INFEASIBLE_STATUSES = ["infeasible"]

# Error epsilon
EPSILON = 1e-4


if __name__ == "__main__":
    # Read the bugs
    bugs = pd.read_csv("data.csv")

    # Filter the errors
    bugs_filtered = bugs[bugs["bug"] == "correctness"]
    bugs_filtered = bugs_filtered.sort_values(by=["error"], ascending=False)

    print(bugs_filtered)
