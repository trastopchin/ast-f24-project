# Future imports
from __future__ import annotations

# Standard library imports
from typing import List, Optional, Generator
from pathlib import Path
import contextlib
import io
import random
from copy import deepcopy
import itertools

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

# def copy_cplex_model(old_cpx: cp.Cplex) -> cp.Cplex:
#     """Clone a cplex model object"""
#     # Check if this works...
#     new_cpx = cp.Cplex()
#     new_cpx.copylp(numcols=old_cpx.variables.get_num(),
#                numrows=old_cpx.linear_constraints.get_num(),
#                objsense=old_cpx.objective.get_sense(),
#                obj=old_cpx.objective.get_linear(),
#                rhs=old_cpx.linear_constraints.get_rhs(),
#                senses=old_cpx.linear_constraints.get_senses(),
#                matbeg=old_cpx.linear_constraints.get_rows()[0],
#                matcnt=old_cpx.linear_constraints.get_rows()[1],
#                matind=old_cpx.linear_constraints.get_rows()[2],
#                matval=old_cpx.linear_constraints.get_coefficients(),
#                lb=old_cpx.variables.get_lower_bounds(),
#                ub=old_cpx.variables.get_upper_bounds())
#     return new_cpx


def solve_cplex_model(cpx: cp.Cplex, debug=False) -> float:
    """
    Solve a CPLEX model and return the objective value.
    """
    # Inspired by: https://github.com/cswaroop/cplex-samples/blob/master/mipex2.py

    cpx.solve()

    # solution.get_status() returns an integer code
    status = cpx.solution.get_status()
    if debug:
        print(cpx.solution.status[status])

    if status == cpx.solution.status.unbounded:
        raise(CplexSolveError("Model is unbounded"))
    if status == cpx.solution.status.infeasible:
        raise(CplexSolveError("Model is infeasible"))
    if status == cpx.solution.status.infeasible_or_unbounded:
        raise(CplexSolveError("Model is infeasible or unbounded"))

    s_method = cpx.solution.get_method()
    s_type   = cpx.solution.get_solution_type()

    if debug:
        print("Solution status = " , status, ":")
        # the following line prints the status as a string
        print(cpx.solution.status[status])
    
    if s_type == cpx.solution.type.none:
        raise(CplexSolveError("No solution available"))

    if debug:
        print("Objective value = " , cpx.solution.get_objective_value())

        x = cpx.solution.get_values(0, cpx.variables.get_num()-1)
        # because we're querying the entire solution vector,
        # x = c.solution.get_values()
        # would have the same effect
        for j in range(cpx.variables.get_num()):
            print("Column %d: Value = %17.10g" % (j, x[j]))

    return cpx.solution.get_objective_value(), status

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
        GRB.TIME_LIMIT: "time_limit"
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
            self.model.Params.timeLimit = time_limit

    def over_time_limit(self):
        self.optimize()
        return self._status_gurobi == "time_limit"

    @staticmethod
    def read_file(filepath: str, time_limit: float = float('inf')) -> MPSFile:
        """Create an MPSFile from a filepath."""
        filename = Path(filepath).name
        with contextlib.redirect_stdout(io.StringIO()):
            gurobi_model = gp.read(filepath)
            cplex_model = cp.Cplex(filepath)

        return MPSFile(filename, gurobi_model, cplex_model, time_limit=time_limit)

    @staticmethod
    def read_files(dir: str, time_limit: float = float('inf')) -> Generator[MPSFile]:
        """Read a list of mps files from a directory."""
        # Iterate in the order of size, small to large
        file_sizes = [(file, file.stat().st_size) for file in Path(dir).iterdir() if file.is_file()]
        file_sizes.sort(key=lambda x: x[1])

        for file, _ in file_sizes:
            mps_file = MPSFile.from_filepath(str(file), time_limit=time_limit)
            yield mps_file

    @staticmethod
    def write_file(filepath: str, mps_file: MPSFile):
        mps_file.model.write(filepath)

    @staticmethod
    def write_files(dir: str, mps_files: List[MPSFile]):
        """Write a list of mps files to a directory."""
        for mps_file in mps_files:
            output_path = Path(dir) / mps_file.filename
            mps_file.model.write(str(output_path))

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
        self.cplex_model = cp.Cplex()

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
            rand_name = str(random.randint(0, 1000000))
            self.gurobi_model.write(f'temp{rand_name}.mps')
            self.cplex_model.read(f'temp{rand_name}.mps')
            Path(f'temp{rand_name}.mps').unlink()



    def append_to_filename(self, to_append: str):
        """Append a string to the filename."""
        stem_and_extension = self.filename.split('.', 1)
        stem = stem_and_extension[0]
        extension = stem_and_extension[1]
        self.filename = stem + to_append + '.' + extension

    def optimize(self, debug=False):
        """Optimize the program and save the objective value."""
        # TODO: do we need to check that self.model.Status == gp.GRB.OPTIMAL?
        if not debug:
            self.gurobi_model.setParam('OutputFlag', 0)
            self.cplex_model.set_results_stream(None)

        if debug:
            print("Optimizing Gurobi model")
        self.gurobi_model.update()
        self.gurobi_model.optimize()
        self._obj_val_gurobi = self.gurobi_model.ObjVal
        self._status_gurobi = MPSFile.gurobi_status_to_string[self.gurobi_model.Status]
        if debug:
            print("Gurobi objective value: ", self._obj_val_gurobi)

        if debug:
            print("Optimizing CPLEX model")
        self._obj_val_cplex, self._status_cplex = solve_cplex_model(self.cplex_model, debug=debug)
        if debug:
            print("CPLEX objective value: ", self._obj_val_cplex)


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

    def check(self, debug=False) -> tuple[bool, str]:
        """Check that the metamorphic relation holds."""
        # Check if the relation holds and get the relation string
        input_solver_types = [Solver.GUROBI, Solver.CPLEX]
        output_solver_types = [Solver.GUROBI, Solver.CPLEX]
        holds_all = True
        for input_solver_type, output_solver_type in itertools.product(input_solver_types, output_solver_types):
            holds, relation_str = self.mutation.metamorphic_relation(
                self.input_files, self.output_file, input_solver_type, output_solver_type
            )

            # Print debug information if the relation doesn't hold
            if not holds and debug:
                print(self.mutation)
                print(f"\tinput: {self.input_files}")
                print(f"\toutput: {self.output_file}")
                print(f"\trelation: {relation_str}")
                print(f"\tinput_solver_type: {input_solver_type}")
                print(f"\toutput_solver_type: {output_solver_type}")
            
            holds_all = holds_all and holds

        return holds, relation_str


class TranslateObjective(MPSMutation):
    """Translates the MIP objective function."""
    translation: int

    def __init__(self, translation=None):
        # TODO: how do we sample objective translations?
        if translation is None:
            translation = np.random.randint(-1000, 1000)
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
        if input_status == "time_limit" and output_status == "time_limit":
            relation = True
            relation_str = "time_limit == time_limit"
        # Otherwise check the relation
        else:
            relation = self.translation + \
                input_obj == output_obj and {input_status} == {output_status}
            relation_str = f"{self.translation} + {input_obj} == {output_obj}, {input_status} == {output_status}"
        relation = self.translation + input_obj == output_obj
        relation_str = f"{self.translation} + {input_obj}  == {output_obj}"

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

        input_obj, input_status = input_file.obj_val() if input_solver_type == Solver.GUROBI else input_file.obj_val_cplex()
        output_obj, output_status = output_file.obj_val() if output_solver_type == Solver.GUROBI else output_file.obj_val_cplex()
        # If both reach the time limit, the relation is "not broken"
        if input_status == "time_limit" and output_status == "time_limit":
            relation = True
            relation_str = "time_limit == time_limit"
        # Otherwise check the relation
        else:
            relation = self.scale * \
                input_obj == output_obj and {input_status} == {output_status}
            relation_str = f"{self.scale} * {input_obj} == {output_obj}, {input_status} == {output_status}"

        return relation, relation_str
