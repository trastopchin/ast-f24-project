import os
from typing import List

from mps_fuzzer import MPSFile, MPSMutation, MPSMetamorphicRelation
from mps_fuzzer import TranslateObjective, ScaleObjective


def set_gurobi_license(filepath: str):
    """Set the gurobi license environment variable."""
    os.environ['GRB_LICENSE_FILE'] = filepath


if __name__ == '__main__':
    set_gurobi_license('./gurobi.lic')

    # Read the seed files
    seed_files = MPSFile.read_files('./input')
    generated_files = list[tuple[MPSFile, MPSMetamorphicRelation]]()

    for file in seed_files:
        print(file)
        if not file.check_consistency_btwn_models(debug=True):
            print("Inconsistent optimal value between Gurobi and CPLEX for: ", file)

    # Iteratiely apply the mutations
    current_files = seed_files
    next_files = list[MPSFile]()
    depth = 4

    for i in range(depth):
        # Translate Objective
        for input_file in current_files:
            translate_objective = TranslateObjective()
            output_file, relation = translate_objective.mutate([input_file])
            holds, relation_str = relation.check(debug=True)
            print(output_file)
            next_files.append(output_file)
            generated_files.append(output_file)

        # Scale Objective
        for input_file in current_files:
            scale_objective = ScaleObjective()
            output_file, relation = scale_objective.mutate([input_file])
            holds, relation_str = relation.check(debug=True)
            print(output_file)
            next_files.append(output_file)
            generated_files.append(output_file)

        current_files = next_files.copy()
        next_files.clear()
