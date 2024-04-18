#!/usr/bin/env python3

from pathlib import Path
import gurobipy as gp

# Get the files
directory = Path('/scripts/collection')
files = [str(file) for file in directory.iterdir() if file.is_file()]
files = sorted(files)

n = 3
for file in files[:n]:

    # Read the model
    model = gp.read(file)

    # Suppress output
    model.setParam('OutputFlag', 0)

    # Optimize the model
    model.optimize()

    # Print the solution
    if model.status == gp.GRB.OPTIMAL:
        print('Optimal objective value:', model.objVal)
        print(model.getVars())

    print()
