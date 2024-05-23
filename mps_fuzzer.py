# Future imports
from __future__ import annotations

# Standard library imports
from typing import List, Optional, Generator
from pathlib import Path
import contextlib
import io
import random
import itertools
import json

# Third-party imports
import numpy as np
import gurobipy as gp
import cplex as cp
from gurobipy import GRB


# Enumerate the possible solver types
class Solver(str):
    GUROBI = "gurobi"
    CPLEX = "cplex"

class CplexSolveError(Exception):
    """Custom exception for Cplex solve errors."""


def solve_cplex_model(cpx: cp.Cplex, debug=False) -> float:
    """
    Solve a CPLEX model and return the objective value.
    """
    # Inspired by: https://github.com/cswaroop/cplex-samples/blob/master/mipex2.py
    def dprint(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    cpx.solve()

    # solution.get_status() returns an integer code
    status = cpx.solution.get_status()

    s_method = cpx.solution.get_method()
    s_type   = cpx.solution.get_solution_type()

    dprint("Solution status = " , status, ":")
    # the following line prints the status as a string
    dprint(cpx.solution.status[status])
    
    if s_type == cpx.solution.type.none:
        dprint("CplexSolveError(No solution available)")
    elif debug:
        print("Objective value = " , cpx.solution.get_objective_value())

        x = cpx.solution.get_values(0, cpx.variables.get_num()-1)
        # because we're querying the entire solution vector,
        # x = c.solution.get_values()
        # would have the same effect
        for j in range(cpx.variables.get_num()):
            print("Column %d: Value = %17.10g" % (j, x[j]))

    cplex_status_to_string = {
        cpx.solution.status.optimal: "optimal",
        cpx.solution.status.feasible: "feasible",
        cpx.solution.status.infeasible: "infeasible",
        cpx.solution.status.infeasible_or_unbounded: "infeasible_or_unbounded",
        cpx.solution.status.unbounded: "unbounded",
        cpx.solution.status.abort_iteration_limit: "abort_iteration_limit",
        cpx.solution.status.abort_time_limit: "abort_time_limit",
        cpx.solution.status.MIP_optimal: "optimal",
        cpx.solution.status.MIP_time_limit_feasible: "time_limit_feasible",
        cpx.solution.status.MIP_time_limit_infeasible: "time_limit_infeasible",
        cpx.solution.status.MIP_dettime_limit_feasible: "dettime_limit_feasible",
        cpx.solution.status.MIP_dettime_limit_infeasible: "dettime_limit_infeasible",
        cpx.solution.status.optimal_tolerance: "optimal_tolerance",
        cpx.solution.status.MIP_abort_feasible: "abort_feasible",
        cpx.solution.status.MIP_abort_infeasible: "abort_infeasible",
        cpx.solution.status.MIP_infeasible: "infeasible",
        # Add more statuses as needed
    }

    dprint("Status: ", cplex_status_to_string[status])

    acceptable_cplex_status_strs = [
        "optimal",
        "time_limit_feasible",
        "optimal_tolerance"
    ]

    return (
        cpx.solution.get_objective_value() 
        if cplex_status_to_string[status] in acceptable_cplex_status_strs else None
        ,
        cplex_status_to_string[status]
    )

class MPSFile:
    """MPSFile class"""
    filename: str
    gurobi_model: gp.Model
    cplex_model: cp.Cplex
    time_limit: float
    _obj_val_gurobi: Optional[float]
    _obj_val_cplex: Optional[float]
    _status_gurobi: str
    _status_cplex: str

    # Gurobi optimization status code to string
    gurobi_status_to_string = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit",
        GRB.INFEASIBLE: "infeasible",
        GRB.INF_OR_UNBD: "inf_or_unbd",
        GRB.UNBOUNDED: "unbounded",
        GRB.CUTOFF: "cutoff",
        GRB.ITERATION_LIMIT: "iteration_limit",
        GRB.NODE_LIMIT: "node_limit",
        GRB.SOLUTION_LIMIT: "solution_limit",
    }



    def __init__(
        self,
        filename: Path,
        gurobi_model: gp.Model,
        cplex_model: Optional[cp.Cplex],
        time_limit: float = float('inf')
    ):
        self.filename = filename
        self.gurobi_model = gurobi_model
        self.cplex_model = cplex_model
        self.time_limit = time_limit
        self._obj_val_gurobi = None
        self._obj_val_cplex = None
        self._status_gurobi = None
        self._status_cplex = None
        self.set_time_limit(time_limit)

    def set_time_limit(self, time_limit: float):
        self.time_limit = time_limit
        with contextlib.redirect_stdout(io.StringIO()):
            self.gurobi_model.Params.TimeLimit = time_limit
            if self.cplex_model is not None:
                self.cplex_model.parameters.timelimit.set(time_limit)


    def over_time_limit(self):
        self.optimize()
        return self._status_gurobi == "time_limit"

    @staticmethod
    def read_file(filepath: str, time_limit: float = float('inf')) -> MPSFile:
        """Create an MPSFile from a filepath."""
        filename = Path(filepath).name
        with contextlib.redirect_stdout(io.StringIO()):
            with contextlib.redirect_stderr(io.StringIO()):
                gurobi_model = gp.read(filepath)
                cplex_model = cp.Cplex(filepath)

        return MPSFile(filename, gurobi_model, cplex_model, time_limit=time_limit)

    @staticmethod
    def read_files(dir: str, time_limit: float = float('inf'), filters=None) -> Generator[MPSFile]:
        """Read a list of mps files from a directory."""
        # Iterate in the order of size, small to large
        file_sizes = [(file, file.stat().st_size) for file in Path(dir).iterdir() if file.is_file() and (filters is None or any(f in file.name for f in filters))]
        file_sizes.sort(key=lambda x: x[1])

        for file, _ in file_sizes:
            mps_file = MPSFile.read_file(str(file), time_limit=time_limit)
            yield mps_file

    @staticmethod
    def write_file(filepath: str, mps_file: MPSFile):
        mps_file.gurobi_model.write(filepath)

    @staticmethod
    def write_files(dir: str, mps_files: List[MPSFile]):
        """Write a list of mps files to a directory."""
        for mps_file in mps_files:
            output_path = Path(dir) / mps_file.filename
            mps_file.gurobi_model.write(str(output_path))

    def copy(self) -> MPSFile:
        """Copy an MPSFile object"""
        self.gurobi_model.update()
        return MPSFile(
            self.filename,
            self.gurobi_model.copy(),
            # copy_cplex_model(self.cplex_model)
            None,
            time_limit=self.time_limit
        )
    
    def refresh_cplex_from_gurobi(self):
        """Refresh the CPLEX model from the Gurobi model."""
        

        # TODO: Maybe compare the approaches

        # First approach: copy the Gurobi model to CPLEX by extracting params
        # self.cplex_model.copylp(numcols=self.gurobi_model.NumVars,
        #                         numrows=self.gurobi_model.NumConstrs,
        #                         objsense=self.gurobi_model.getAttr(gp.GRB.Attr.ModelSense),
        #                         obj=self.gurobi_model.getAttr(gp.GRB.Attr.Obj),
        #                         rhs=self.gurobi_model.getAttr(gp.GRB.Attr.RHS),
        #                         senses=self.gurobi_model.getAttr(gp.GRB.Attr.Sense),
        #                         matbeg=self.gurobi_model.getAttr(gp.GRB.Attr.VBasis),
        #                         matcnt=self.gurobi_model.getAttr(gp.GRB.Attr.VBasis),
        #                         matind=self.gurobi_model.getAttr(gp.GRB.Attr.VBasis),
        #                         matval=self.gurobi_model.getAttr(gp.GRB.Attr.VBasis),
        #                         lb=self.gurobi_model.getAttr(gp.GRB.Attr.LB),
        #                         ub=self.gurobi_model.getAttr(gp.GRB.Attr.UB))

        # Second approach: copy the Gurobi model to CPLEX by reading and writing to a fake MPS file
        with contextlib.redirect_stdout(io.StringIO()):
            self.cplex_model = cp.Cplex()
            self.cplex_model.set_warning_stream(None)
            rand_name = str(random.randint(0, 1000000))
            self.gurobi_model.write(f'temp{rand_name}.mps')
            self.cplex_model.read(f'temp{rand_name}.mps')
            Path(f'temp{rand_name}.mps').unlink()
            self.cplex_model.parameters.timelimit.set(self.time_limit)



    def append_to_filename(self, to_append: str):
        """Append a string to the filename."""
        stem_and_extension = self.filename.split('.', 1)
        stem = stem_and_extension[0]
        extension = stem_and_extension[1]
        self.filename = stem + to_append + '.' + extension

    def optimize(self, debug=False):
        """Optimize the program and save the objective value."""
        def dprint(*args, **kwargs):
            if debug:
                print(*args, **kwargs)

        # TODO: do we need to check that self.model.Status == gp.GRB.OPTIMAL?
        if not debug:
            self.gurobi_model.setParam('OutputFlag', 0)
            self.cplex_model.set_results_stream(None)

        dprint("Optimizing Gurobi model")
        self.gurobi_model.update()
        self.gurobi_model.optimize()
        self._status_gurobi = MPSFile.gurobi_status_to_string[self.gurobi_model.Status]
        # print(self.gurobi_model.SolCount, self._status_gurobi)
        self._obj_val_gurobi = self.gurobi_model.ObjVal if self.gurobi_model.SolCount > 0 else None

        if self._status_gurobi == "time_limit":
            if self.gurobi_model.SolCount > 0:
                self._status_gurobi = "time_limit_feasible"
            else:
                self._status_gurobi = "time_limit_infeasible"


        dprint("Gurobi status: ", self._status_gurobi)
        dprint("Gurobi objective value: ", self._obj_val_gurobi)

        dprint("Optimizing CPLEX model")

        self._obj_val_cplex, self._status_cplex = solve_cplex_model(self.cplex_model, debug=debug)
        dprint("CPLEX objective value: ", self._obj_val_cplex)


    def obj_val_gurobi(self, debug=False):
        """Lazy compute the Gurobi objective value."""
        if self._obj_val_gurobi == None:
            self.optimize(debug=debug)
        elif debug:
            print("Caching Gurobi objective value")
        return self._obj_val_gurobi, self._status_gurobi

    def obj_val_cplex(self, debug=False):
        """Lazy compute the CPLEX objective value."""
        if self._obj_val_cplex == None:
            self.optimize(debug=debug)
        elif debug:
            print("Caching CPLEX objective value")
        return self._obj_val_cplex, self._status_cplex

    def __repr__(self) -> str:
        """Return the filename."""
        return self.filename
    
    def check_consistency_btwn_models(self, debug=False) -> bool:
        gurobi_val = self.obj_val_gurobi(debug=debug)
        cplex_val = self.obj_val_cplex(debug=debug)
        return gurobi_val == cplex_val


POSITIVE_STATUSES = ["optimal", "time_limit_feasible"]
NEGATIVE_STATUSES = ["infeasible", "time_limit_infeasible"]
def status_cmp(status1: str, status2: str) -> bool:
    """Compare two solver statuses."""
    if status1 == status2:
        return True
    elif status1 in POSITIVE_STATUSES and status2 in POSITIVE_STATUSES:
        return True
    elif status1 in NEGATIVE_STATUSES and status2 in NEGATIVE_STATUSES:
        return True

    return False


class MPSMutation:
    """Represents a mutation of an MPSFile."""

    def mutate(self, input: List[MPSFile]) -> tuple[MPSFile, Optional[MPSMetamorphicRelation]]:
        pass

    def metamorphic_relation(self, input_files: List[MPSFile], output_file: MPSFile) -> tuple[bool, str]:
        pass


class MPSMetamorphicRelation:
    """Represents a metamorphic relation between a list of input MPSFiles and an output MPSFile."""
    input_files: List[MPSFile]
    output_file: MPSFile
    mutation: MPSMutation

    def __init__(self, input_files: List[MPSFile], output_file: MPSFile, mutation: MPSMutation):
        self.input_files = input_files
        self.output_file = output_file
        self.mutation = mutation

    def check(self, solver: Solver, debug=False) -> tuple[bool, str]:
        """Check that the metamorphic relation holds."""
        input_solver = solver
        output_solver = solver
        # Check if the relation holds and get the relation string
        holds, relation_str = self.mutation.metamorphic_relation(
            self.input_files, self.output_file, input_solver, output_solver
        )

        # Print debug information if the relation doesn't hold
        if not holds and debug:
            print(self.mutation)
            print(f"\tinput: {self.input_files}")
            print(f"\toutput: {self.output_file}")
            print(f"\trelation: {relation_str}")
            print(f"\tinput_solver: {input_solver}")
            print(f"\toutput_solver_type: {output_solver}")
        
        return holds, relation_str


class TranslateObjective(MPSMutation):
    """Translates the MIP objective function."""
    translation: int

    def __init__(self, translation=None):
        # TODO: how do we sample objective translations?
        if translation is None:
            translation = np.random.randint(-10000, 10000)
            # TODO: this breaks the metamorphic relation
            # translation = np.random.randint(-100000, 100000)
        self.translation = translation

    def mutate(self, input_files: List[MPSFile]) -> tuple[MPSFile, Optional[MPSMetamorphicRelation]]:
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        # Copy the MPSFile
        output_file = input_file.copy()
        output_file.append_to_filename('_TO')

        # Translate the objective
        objective = output_file.gurobi_model.getObjective()
        output_file.gurobi_model.setObjective(self.translation + objective)
        output_file.refresh_cplex_from_gurobi()

        # Create the metamorphic relation
        metamorphic_relation = MPSMetamorphicRelation(
            input_files, output_file, self)

        # Return the output file and the metamorphic relation
        return output_file, metamorphic_relation

    def metamorphic_relation(
            self,
            input_files: List[MPSFile],
            output_file: MPSFile,
            input_solver_type: Solver = Solver.GUROBI,
            output_solver_type: Solver = Solver.GUROBI
    ) -> tuple[bool, str]:
        """Metamorphic relation."""
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        input_obj, input_status = input_file.obj_val_gurobi() if input_solver_type == Solver.GUROBI else input_file.obj_val_cplex()
        output_obj, output_status = output_file.obj_val_gurobi() if output_solver_type == Solver.GUROBI else output_file.obj_val_cplex()
        # If both reach the time limit, the relation is "not broken"
        if status_cmp(input_status, output_status) and input_status not in POSITIVE_STATUSES:
            relation = input_status == output_status
            relation_str = f"{input_status} == {output_status}"
        # Otherwise check the relation
        else:
            relation = status_cmp(input_status, output_status) and (self.translation + input_obj) == output_obj
            relation_str = f"{self.translation} + {input_obj} == {output_obj}, {input_status} == {output_status}"

        return relation, relation_str


class ScaleObjective(MPSMutation):
    """Scale the MIP objective function."""
    scale: int

    def __init__(self, scale: int = None):
        # TODO: how do we sample objective scales?
        if scale is None:
            scale = np.random.randint(1, 10)
        self.scale = scale

    def mutate(self, input_files: List[MPSFile]) -> tuple[MPSFile, Optional[MPSMetamorphicRelation]]:
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        # Copy the MPSFile
        output_file = input_file.copy()
        output_file.append_to_filename('_TS')

        # Scale the objective
        objective = output_file.gurobi_model.getObjective()
        output_file.gurobi_model.setObjective(self.scale * objective)

        output_file.refresh_cplex_from_gurobi()

        # Create the metamorphic relation
        metamorphic_relation = MPSMetamorphicRelation(
            input_files, output_file, self)

        # Return the output file and the metamorphic relation
        return output_file, metamorphic_relation

    def metamorphic_relation(
            self,
            input_files: List[MPSFile],
            output_file: MPSFile,
            input_solver_type: Solver = Solver.GUROBI,
            output_solver_type: Solver = Solver.GUROBI
    ) -> tuple[bool, str]:
        """Metamorphic relation."""
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        input_obj, input_status = input_file.obj_val_gurobi() if input_solver_type == Solver.GUROBI else input_file.obj_val_cplex()
        output_obj, output_status = output_file.obj_val_gurobi() if output_solver_type == Solver.GUROBI else output_file.obj_val_cplex()
        # If both reach the time limit, the relation is "not broken"
        if status_cmp(input_status, output_status) and input_status not in POSITIVE_STATUSES:
            relation = input_status == output_status
            relation_str = f"{input_status} == {output_status}"
        # Otherwise check the relation
        else:
            relation = status_cmp(input_status, output_status) and (self.scale * input_obj) == output_obj
            relation_str = f"{self.scale} * {input_obj} == {output_obj}, {input_status} == {output_status}"

        return relation, relation_str



class Result:
    
    def __init__(self, data: dict):
        self.data = data

    def __repr__(self):
        return json.dumps(self.data)
    
    @staticmethod
    def optimizer_consistency(file: MPSFile, bug: bool) -> Result:
        # Record the generated result
        result = {}
        result['type'] = 'consistency'
        result['bug'] = bug
        result['file'] = file.filename
        # To help filter previously processed files
        result['relation'] = 'Consistency'
        obj_val_gurobi, status_gurobi = file.obj_val_gurobi()
        obj_val_cplex, status_cplex = file.obj_val_cplex()
        result['obj_val_gurobi'] = obj_val_gurobi
        result['status_gurobi'] = status_gurobi
        result['obj_val_cplex'] = obj_val_cplex
        result['status_cplex'] = status_cplex
        return Result(result)
    
    @staticmethod
    def metamorphic_relation(relation: MPSMetamorphicRelation, holds: bool, relation_str: str, solver: Solver) -> Result:
        # Record the generated result
        result = {}
        result['type'] = 'metamorphic'
        result['bug'] = not holds
        # Assuming all of our mutations have exactly one input file!
        result['file'] = relation.input_files[0].filename
        # Just in case?
        if (len(relation.input_files) > 1):
            result['input_files'] = sorted([file.filename for file in relation.input_files])
        result['output_file'] = relation.output_file.filename
        result['relation'] = type(relation.mutation).__name__
        result['relation_str'] = relation_str
        result['solver'] = solver
        return Result(result)