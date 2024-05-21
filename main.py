import os
from typing import List

from mps_fuzzer import MPSFile, MPSMutation, MPSMetamorphicRelation
from mps_fuzzer import TranslateObjective, ScaleObjective
from tqdm import tqdm

from pathlib import Path


def set_gurobi_license(filepath: str):
    """Set the gurobi license environment variable."""
    os.environ['GRB_LICENSE_FILE'] = filepath


# No need to do this since we sort by file size
# def remove_over_time_limit(mps_files: List[MPSFile], num_files: int):
#     valid_files = []
#     for mps_file in tqdm(mps_files, total=num_files,
#                          desc="Remove over time limit"):
#         if not mps_file.over_time_limit():
#             valid_files.append(mps_file)
#     return valid_files


if __name__ == '__main__':
    set_gurobi_license('./gurobi.lic')

    # fast_inputs = ['10teams.mps.gz', '30_70_45_05_100.mps.gz', '30_70_45_095_100.mps.gz', '30_70_45_095_98.mps.gz', '30n20b8.mps.gz']

    # Read the seed files
    INPUT_SEED_DIR='./input'
    seed_files = MPSFile.read_files(INPUT_SEED_DIR, time_limit=20)
    # num_files = sum(1 for file in Path(INPUT_SEED_DIR).iterdir() if file.is_file())
    # seed_files = remove_over_time_limit(seed_files, num_files)
    generated_files = list[tuple[MPSFile, MPSMetamorphicRelation]]()

    for file in seed_files:
        print(file)
        if not file.check_consistency_btwn_models(debug=True):
            print("Inconsistent optimal value between Gurobi and CPLEX for: ", file)

        # Iteratively apply the mutations
        current_files = [file]
        next_files = list[MPSFile]()
        depth = 1

        for i in range(depth):
            desc = f"Iteration {i+1}/{depth}"

            # Translate Objective
            for input_file in tqdm(current_files, total=len(current_files),
                                desc=f"{desc}; Translate Objective"):
                translate_objective = TranslateObjective()
                output_file, relation = translate_objective.mutate([input_file])
                holds, relation_str = relation.check(debug=True)
                if not output_file.over_time_limit():
                    next_files.append(output_file)
                    generated_files.append(output_file)

            # Scale Objective
            for input_file in tqdm(current_files, total=len(current_files),
                                desc=f"{desc}; Scale Objective"):
                scale_objective = ScaleObjective()
                output_file, relation = scale_objective.mutate([input_file])
                holds, relation_str = relation.check(debug=True)
                if not output_file.over_time_limit():
                    next_files.append(output_file)
                    generated_files.append(output_file)

            current_files = next_files.copy()
            next_files.clear()

    MPSFile.write_files('./output', generated_files)
    print("Wrote generated MPS files!")
