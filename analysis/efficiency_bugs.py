from analysis import create_dataframe

df = create_dataframe()

df = df[df.bug == "efficiency"]

consistency_bugs = df[df.type == "consistency"].drop(columns=["type", "bug"])
metamorphic_bugs = df[df.type == "metamorphic"].drop(columns=["type", "bug"])


# Consistency bugs
print("CONSISTENCY BUGS")
# Go over every row of the consistency bugs and determine if CPLEX performed better or GUROBI
gurobi_optimality_wins = 0
cplex_optimality_wins = 0
gurobi_time_wins = 0
cplex_time_wins = 0
gurobi_optimality_against_infeasiblity_wins = 0
cplex_optimality_against_infeasiblity_wins = 0
serious_bugs = 0
both_time_limit_feasible = 0
for row in consistency_bugs.itertuples():
    # print(row.status1)  # GUROBI
    # print(row.status2)  # CPLEX
    # print(row.error)  # GUROBI - CPLEX

    gurobi_status = row.status1
    cplex_status = row.status2

    if gurobi_status in ["optimal", "optimal_tolerance"] and cplex_status in [
        "time_limit_feasible"
    ]:
        gurobi_optimality_wins += 1

    elif cplex_status in ["optimal", "optimal_tolerance"] and gurobi_status in [
        "time_limit_feasible"
    ]:
        cplex_optimality_wins += 1
    elif gurobi_status in ["optimal", "optimal_tolerance"] and cplex_status in [
        "time_limit_infeasible"
    ]:
        gurobi_optimality_against_infeasiblity_wins += 1
        print(row)
    elif gurobi_status in ["time_limit_feasible"] and cplex_status in [
        "time_limit_feasible"
    ]:
        both_time_limit_feasible += 1
    else:
        print(gurobi_status, cplex_status)
        print("Needs more cases")
        assert False

print("Gurobi optimality wins:", gurobi_optimality_wins)
print("Cplex optimality wins:", cplex_optimality_wins)
print("Gurobi time wins:", gurobi_time_wins)
print("Cplex time wins:", cplex_time_wins)
print(
    "Gurobi optimality against infeasibility wins:",
    gurobi_optimality_against_infeasiblity_wins,
)
print("Both time limit feasible:", both_time_limit_feasible)

print("METAMORPHIC BUGS")
# Metamorphic bugs
metamorphic_bugs_gurobi = metamorphic_bugs[metamorphic_bugs.solver == "gurobi"].drop(
    columns=["solver"]
)
metamorphic_bugs_cplex = metamorphic_bugs[metamorphic_bugs.solver == "cplex"].drop(
    columns=["solver"]
)

metamorphic_summary_str = ""
for solver in ["gurobi", "cplex"]:
    print(f"{solver} metamorphic bugs")
    for mutation_op in ["TranslateObjective", "ScaleObjective"]:
        print("Mutation operation:", mutation_op)
        pre_mutation_wins = 0
        post_mutation_wins = 0
        both_time_limit_feasible = 0
        pre_mutation_optimality_against_infeasibility_wins = 0
        pre_mutation_feasibility_against_infeasibility_wins = 0

        mbdf = metamorphic_bugs_gurobi if solver == "gurobi" else metamorphic_bugs_cplex
        for row in mbdf[mbdf.mutation_type == mutation_op].itertuples():
            pre_mutation_status = row.status1
            post_mutation_status = row.status2

            if pre_mutation_status in [
                "optimal",
                "optimal_tolerance",
            ] and post_mutation_status in ["time_limit_feasible"]:
                pre_mutation_wins += 1
            elif post_mutation_status in [
                "optimal",
                "optimal_tolerance",
            ] and pre_mutation_status in ["time_limit_feasible"]:
                post_mutation_wins += 1
            elif pre_mutation_status in [
                "time_limit_feasible"
            ] and post_mutation_status in ["time_limit_feasible"]:
                both_time_limit_feasible += 1
            elif pre_mutation_status in [
                "optimal",
                "optimal_tolerance",
            ] and post_mutation_status in ["time_limit_infeasible"]:
                print(row)
                pre_mutation_optimality_against_infeasibility_wins += 1
            elif pre_mutation_status in [
                "time_limit_feasible"
            ] and post_mutation_status in ["time_limit_infeasible"]:
                pre_mutation_feasibility_against_infeasibility_wins += 1
            else:
                print(pre_mutation_status, post_mutation_status)
                print("Needs more cases")
                assert False
        print("Pre mutation wins:", pre_mutation_wins)
        print("Post mutation wins:", post_mutation_wins)
        print("Both time limit feasible:", both_time_limit_feasible)
        print(
            "Pre mutation optimality against infeasibility wins:",
            pre_mutation_optimality_against_infeasibility_wins,
        )
        print(
            "Pre mutation feasibility against infeasibility wins:",
            pre_mutation_feasibility_against_infeasibility_wins,
        )

        pre_wins = (
            pre_mutation_wins
            + pre_mutation_optimality_against_infeasibility_wins
            + pre_mutation_feasibility_against_infeasibility_wins
        )
        post_wins = post_mutation_wins
        metamorphic_summary_str += f"SUMMARY for {solver} {mutation_op}:\n"
        metamorphic_summary_str += f"Pre wins: {pre_wins}\n"
        metamorphic_summary_str += f"Post wins: {post_wins}\n"
        metamorphic_summary_str += "\n"

print(metamorphic_summary_str)
