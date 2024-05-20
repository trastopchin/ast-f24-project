# Future imports
from __future__ import annotations

# Standard library imports
from typing import List, Optional
from pathlib import Path
import contextlib
import io

# Third-party imports
import numpy as np
import gurobipy as gp
from gurobipy import GRB


class MPSFile:
    """MPSFile class"""
    filename: str
    model: gp.Model
    time_limit: float
    _obj_val: Optional[float]
    _status: str

    # Gurobi optimization status code to string
    gurobi_status_to_string = {
        GRB.OPTIMAL: "optimal",
        GRB.TIME_LIMIT: "time_limit"
    }

    def __init__(self, filename: Path, model: gp.Model, time_limit: float = float('inf')):
        self.filename = filename
        self.model = model
        self.time_limit = time_limit
        self._obj_val = None
        self._status = None
        self.set_time_limit(time_limit)

    def set_time_limit(self, time_limit: float):
        self.time_limit = time_limit
        with contextlib.redirect_stdout(io.StringIO()):
            self.model.Params.timeLimit = time_limit

    def over_time_limit(self):
        self.optimize()
        return self._status == "time_limit"

    @staticmethod
    def read_file(filepath: str, time_limit: float = float('inf')) -> MPSFile:
        """Create an MPSFile from a filepath."""
        filename = Path(filepath).name
        model = None
        with contextlib.redirect_stdout(io.StringIO()):
            model = gp.read(filepath)

        return MPSFile(filename, model, time_limit=time_limit)

    @staticmethod
    def read_files(dir: str, time_limit: float = float('inf')) -> List[MPSFile]:
        """Read a list of mps files from a directory."""
        mps_files = []
        for file in Path(dir).iterdir():
            if file.is_file():
                mps_file = MPSFile.read_file(str(file), time_limit=time_limit)
                mps_files.append(mps_file)
        return mps_files

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
        self.model.update()
        return MPSFile(self.filename, self.model.copy(), time_limit=self.time_limit)

    def append_to_filename(self, to_append: str):
        """Append a string to the filename."""
        stem_and_extension = self.filename.split('.', 1)
        stem = stem_and_extension[0]
        extension = stem_and_extension[1]
        self.filename = stem + to_append + '.' + extension

    def _optimize(self):
        """Optimize the program and save the objective value."""
        # TODO: do we need to check that self.model.Status == gp.GRB.OPTIMAL?
        self.model.setParam('OutputFlag', 0)
        self.model.update()
        self.model.optimize()
        self._obj_val = self.model.ObjVal
        self._status = MPSFile.gurobi_status_to_string[self.model.Status]

    def optimize(self):
        """Lazy optimize the model."""
        if self._obj_val == None:
            self._optimize()
        return self._obj_val, self._status

    def __repr__(self) -> str:
        """Return the filename."""
        return self.filename


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
        holds, relation_str = self.mutation.metamorphic_relation(
            self.input_files, self.output_file)

        # Print debug information if the relatino doesn't hold
        if not holds and debug:
            print(self.mutation)
            print(f"\tinput: {self.input_files}")
            print(f"\toutput: {self.output_file}")
            print(f"\trelation: {relation_str}")

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
        objective = output_file.model.getObjective()
        output_file.model.setObjective(self.translation + objective)

        # Create the metamorphic relation
        metamorphic_relation = MPSMetamorphicRelation(
            input_files, output_file, self)

        # Return the output file and the metamorphic relation
        return output_file, metamorphic_relation

    def metamorphic_relation(self, input_files: List[MPSFile], output_file: MPSFile) -> tuple[bool, str]:
        """Metamorphic relation."""
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        input_obj, input_status = input_file.optimize()
        output_obj, output_status = output_file.optimize()

        # If both reach the time limit, the relation is "not broken"
        if input_status == "time_limit" and output_status == "time_limit":
            relation = True
            relation_str = "time_limit == time_limit"
        # Otherwise check the relation
        else:
            relation = self.translation + \
                input_obj == output_obj and {input_status} == {output_status}
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
        objective = output_file.model.getObjective()
        output_file.model.setObjective(self.scale * objective)

        # Create the metamorphic relation
        metamorphic_relation = MPSMetamorphicRelation(
            input_files, output_file, self)

        # Return the output file and the metamorphic relation
        return output_file, metamorphic_relation

    def metamorphic_relation(self, input_files: List[MPSFile], output_file: MPSFile) -> tuple[bool, str]:
        """Metamorphic relation."""
        # Assert there is one input file
        assert (len(input_files) == 1)
        input_file = input_files[0]

        input_obj, input_status = input_file.optimize()
        output_obj, output_status = output_file.optimize()

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
