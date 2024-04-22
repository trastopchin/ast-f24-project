from __future__ import annotations
from typing import Optional, List
import os
import gurobipy as gp
import numpy as np
import contextlib
import io
import queue
import random

# Forward references


class MPS:
    pass


class Mutation:
    pass


def set_gurobi_license(filepath: str):
    """Set the gurobi license environment variable."""
    os.environ['GRB_LICENSE_FILE'] = filepath


class MPS:
    """MPS class"""
    model: Optional[gp.Model]
    obj_val: Optional[float]
    solution: Optional[np.ndarray]
    mutations: List[Mutation]

    def __init__(self):
        """Empty constructor."""
        self.model = None
        self.obj_val = None
        self.mutations = []

    def optimize(self):
        """Optimize the program and save the objective value"""

        if self.model == None:
            print('Model not initialized.')
            return

        self.model.setParam('OutputFlag', 0)
        self.model.optimize()
        if self.model.Status == gp.GRB.OPTIMAL:
            self.obj_val = float(self.model.ObjVal)
            self.solution = np.zeros(self.model.NumVars)
            for i in range(self.model.NumVars):
                self.solution[i] = self.model.getVars()[i].X
        else:
            self.obj_val = None
            self.solution = None

    @ staticmethod
    def from_model(model: gp.Model):
        """Create an MPS object from a gp.Model."""
        mps = MPS()
        mps.model = model.copy()
        mps.optimize()
        return mps

    @ staticmethod
    def from_filepath(filepath: str):
        """Create an MPS object from a filepath"""

        # Read the mps file
        model = None
        with contextlib.redirect_stdout(io.StringIO()):
            model = gp.read(filepath)

        return MPS.from_model(model)

    def copy(self):
        """Copy an MPS object"""
        mps = MPS()
        mps.model = self.model.copy()
        mps.obj_val = self.obj_val
        mps.solution = self.solution.copy()
        mps.mutations = self.mutations.copy()
        return mps

    def __repr__(self) -> str:
        """Print the number of constraints, the number of variables, and the number of mutations"""
        return f"(Num Mutations: {len(self.mutations)}, Mutations: {str(self.mutations)})"

    @ staticmethod
    def _linexpr_to_ndarray(linexpr: gp.LinExpr) -> np.ndarray:
        """Convert a gp.LinExpr to a np.ndarray"""
        n = linexpr.size()
        ndarray = np.zeros(n + 1)
        ndarray[0] = linexpr.getConstant()
        for i in range(n):
            ndarray[i+1] = linexpr.getCoeff(i)
        return ndarray

    @ staticmethod
    def _ndarray_to_linexpr(model: gp.Model, ndarray: np.ndarray) -> gp.LinExpr:
        """Convert a np.ndarray to a gp.LinExpr"""
        linexpr = gp.LinExpr()
        linexpr.addConstant(ndarray[0])
        linexpr.addTerms(ndarray[1:], model.getVars())
        return linexpr


class Mutation:
    """Mutation base class."""

    def mutate(self, mps: MPS):
        pass


class ScaleObjective(Mutation):
    """Scale objective mutation"""

    def __init__(self, scale: int):
        self.scale = np.random.randint(-scale, scale)

    def __repr__(self):
        return f"{self.scale} * objective"

    def mutate(self, mps: MPS):
        # Perform the mutation
        mps_new = mps.copy()
        mps_new.mutations.append(self)
        objective_linexpr = mps_new.model.getObjective()
        objective_ndarray = MPS._linexpr_to_ndarray(objective_linexpr)
        objective_ndarray_new = self.scale * objective_ndarray
        objective_linexpr_new = MPS._ndarray_to_linexpr(
            mps_new.model, objective_ndarray_new)
        mps_new.model.setObjective(objective_linexpr_new)
        mps_new.optimize()

        # Check the metamorphic relationship
        objective_relation = mps_new.obj_val == self.scale * mps.obj_val
        if not objective_relation:
            print("Objective relation doesn't hold!")
            print(f"Expected objective: {self.scale * mps.obj_val}")
            print(f"Actual objective: {mps_new.obj_val}")
        # solution_relation = np.array_equal(mps_new.solution, mps.solution)
        # if not solution_relation:
        #     print("Solution relation doesn't hold!")

        return mps_new


class TranslateObjective(Mutation):
    """Translate objective mutation"""

    def __init__(self, translation: int):
        self.translation = np.random.randint(-translation, translation)

    def __repr__(self):
        return f"{self.translation} + objective"

    def mutate(self, mps: MPS):

        # Perform the mutation
        mps_new = mps.copy()
        mps_new.mutations.append(self)
        objective_linexpr = mps_new.model.getObjective()
        objective_ndarray = MPS._linexpr_to_ndarray(objective_linexpr)
        objective_ndarray_new = objective_ndarray.copy()
        objective_ndarray_new[0] += self.translation
        objective_linexpr_new = MPS._ndarray_to_linexpr(
            mps_new.model, objective_ndarray_new)
        mps_new.model.setObjective(objective_linexpr_new)
        mps_new.optimize()

        # Check the metamorphic relationship
        objective_relation = mps_new.obj_val == self.translation + mps.obj_val
        if not objective_relation:
            print("Objective relation doesn't hold!")
            print(f"Expected objective: {self.translation + mps.obj_val}")
            print(f"Actual objective: {mps_new.obj_val}")
        # solution_relation = np.array_equal(mps_new.solution, mps.solution)
        # if not solution_relation:
        #     print("Solution relation doesn't hold!")
        return mps_new


class TranslateVariables(Mutation):
    """Translate variables mutation"""

    def __init__(self, translation: int):
        self.initialzed = False
        self.translation = translation

    def __repr__(self):
        return f"{str(self.translation)} + variables"

    def mutate(self, mps: MPS):
        if not self.initialized:
            self.translation = np.random.randint(
                -self.translation, self.translation, shape=(mps.model.NumVars,))
            self.initialzed = True

        # Perform the mutation
        mps_new = mps.copy()
        mps_new.mutations.append(self)
        objective_linexpr = mps_new.model.getObjective()
        objective_ndarray = MPS._linexpr_to_ndarray(objective_linexpr)
        objective_ndarray_new = objective_ndarray.copy()
        objective_ndarray_new += self.translation
        objective_linexpr_new = MPS._ndarray_to_linexpr(
            mps_new.model, objective_ndarray_new)
        mps_new.model.setObjective(objective_linexpr_new)
        mps_new.optimize()

        # Check the metamorphic relationship
        objective_relation = mps_new.obj_val == mps.obj_val
        if not objective_relation:
            print("Objective relation doesn't hold!")
            print(f"Expected objective: {mps.obj_val}")
            print(f"Actual objective: {mps_new.obj_val}")
        # solution_relation = np.array_equal(mps_new.solution, mps.solution)
        # if not solution_relation:
        #     print("Solution relation doesn't hold!")
        return mps_new


if __name__ == '__main__':
    # Gurobi license path
    set_gurobi_license('./gurobi.lic')

    # Seed filepath
    filepath = './10teams.mps.gz'
    seed = MPS.from_filepath(filepath)

    # Depth first search
    depth = 10
    generated = []
    queue = queue.Queue()
    queue.put(seed)

    while not queue.empty():
        current = queue.get()

        if len(current.mutations) >= depth:
            continue

        mutations = [ScaleObjective(10), TranslateObjective(
            1000), TranslateObjective(1000)]
        mutation = random.choice(mutations)
        mutated = mutation.mutate(current)
        print(mutated)
        queue.put(mutated)
