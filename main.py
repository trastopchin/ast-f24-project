import os
from typing import List
from pathlib import Path

from mps_fuzzer import MPSFile, MPSMutation, MPSMetamorphicRelation
from mps_fuzzer import TranslateObjective, ScaleObjective
from mps_fuzzer import OptimizerBug
import json
from tqdm import tqdm


def set_gurobi_license(filepath: str):
    """Set the gurobi license environment variable."""
    os.environ['GRB_LICENSE_FILE'] = filepath

def append_to_results(data: str):
    filepath = "./results.txt"
    with open(filepath, 'a') as file:
        file.write(data + "\n")

if __name__ == '__main__':
    set_gurobi_license('./gurobi.lic')

    # Read the seed files
    INPUT_SEED_DIR='./input'
    filters = None
    # filters = ['noswot'] # Comment this line to include all files
    seed_files = MPSFile.read_files(INPUT_SEED_DIR, time_limit=20, filters=filters)
    n_seed_files = len([file for file in Path(INPUT_SEED_DIR).iterdir() if file.is_file() and (filters is None or any(f in file.name for f in filters))])
    generated_files = list[tuple[MPSFile, MPSMetamorphicRelation]]()
    generated_bugs = list[dict]()

    for file in (pbar := tqdm(seed_files, total=n_seed_files)):
        pbar.set_description(f"Processing {file}")
        # Consistency bug
        if not file.check_consistency_btwn_models(debug=False):
            # print("Inconsistent optimal value between Gurobi and CPLEX for: ", file)
            bug = OptimizerBug.consistency_bug(file)
            generated_bugs.append(bug)
            data = json.dumps(bug)
            #print(data)
            append_to_results(data)
        # Consistency bug

        # Iteratively apply the mutations
        current_files = [file]
        next_files = list[MPSFile]()
        depth = 1

        for i in range(depth):
            # desc = f"Iteration {i+1}/{depth}"

            # Translate Objective
            for input_file in current_files:
                translate_objective = TranslateObjective()
                output_file, relation = translate_objective.mutate([input_file])
                holds, relation_str = relation.check(debug=False)
                if not output_file.over_time_limit():
                    next_files.append(output_file)
                    generated_files.append(output_file)
                    # Metamorphic bug
                    if not holds:
                        bug = OptimizerBug.metamorphic_bug(relation)
                        generated_bugs.append(bug)
                        data = json.dumps(bug)
                        # print(data)
                        append_to_results(data)

            # Scale Objective
            for input_file in current_files:
                scale_objective = ScaleObjective()
                output_file, relation = scale_objective.mutate([input_file])
                holds, relation_str = relation.check(debug=False)
                if not output_file.over_time_limit():
                    next_files.append(output_file)
                    generated_files.append(output_file)
                    # Metamorphic bug
                    if not holds:
                        bug = OptimizerBug.metamorphic_bug(relation)
                        generated_bugs.append(bug)
                        data = json.dumps(bug)
                        # print(data)
                        append_to_results(data)

            current_files = next_files.copy()
            next_files.clear()

    # Write the generated files
    MPSFile.write_files('./output', generated_files)
    print("Wrote generated MPS files!")
        
