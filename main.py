import os
from typing import List
from pathlib import Path

from mps_fuzzer import MPSFile, MPSMutation, MPSMetamorphicRelation
from mps_fuzzer import TranslateObjective, ScaleObjective
from mps_fuzzer import Result
import json
from tqdm import tqdm


RESULTS_FILEPATH = "./results.txt"

def set_gurobi_license(filepath: str):
    """Set the gurobi license environment variable."""
    os.environ['GRB_LICENSE_FILE'] = filepath

def append_to_results(data: str):
    with open(RESULTS_FILEPATH, 'a') as file:
        file.write(data + "\n")

if __name__ == '__main__':
    set_gurobi_license('./gurobi.lic')

    # Read the seed files
    INPUT_SEED_DIR='./input'
    filters = None
    # filters = ['noswot'] # Comment this line to include all files
    seed_files = MPSFile.read_files(INPUT_SEED_DIR, time_limit=20, filters=filters)
    n_seed_files = len([file for file in Path(INPUT_SEED_DIR).iterdir() if file.is_file() and (filters is None or any(f in file.name for f in filters))])
    
    # Read the processed files
    processed = set[tuple[str, str]]()
    try:
        with open(RESULTS_FILEPATH, 'r') as file:
            for line in file:
                result = json.loads(line)
                processed.add((result['file'], result['relation']))
    except FileNotFoundError:
        pass

    # Process each unprocessed file
    for file in (pbar := tqdm(seed_files, total=n_seed_files)):
        
        if (file.filename, "Consistency") not in processed:
            # Update progress bar description
            pbar.set_description(f"Processing {file}. Consistency check")
            
            # Check for consistency between optimizers
            bug = not file.check_consistency_btwn_models(debug=False)
            
            # Log the result
            result = Result.optimizer_consistency(file, bug)
            append_to_results(str(result))


        if (file.filename, "TranslateObjective") not in processed:
            # Update progress bar description
            pbar.set_description(f"Processing {file}. Translate objective")

            # Translate Objective
            translate_objective = TranslateObjective()
            output_file, relation = translate_objective.mutate([file])
            
            # Check that metamorphic relation holds
            holds, relation_str = relation.check(debug=False)
            
            # Log the result
            result = Result.metamorphic_relation(relation, holds, relation_str)
            append_to_results(str(result))
        
        if (file.filename, "ScaleObjective") not in processed:
            # Update progress bar description
            pbar.set_description(f"Processing {file}. Scale objective")

            # Scale Objective
            scale_objective = ScaleObjective()
            output_file, relation = scale_objective.mutate([file])
            
            # Check that metamorphic relation holds
            holds, relation_str = relation.check(debug=False)
            
            # Log the result
            result = Result.metamorphic_relation(relation, holds, relation_str)
            append_to_results(str(result))