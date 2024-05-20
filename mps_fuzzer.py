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


class MPSFile:
    """MPSFile class"""
    filename: str
    model: gp.Model
    _obj_val: Optional[float]

    def __init__(self, filename: Path, model: gp.Model):
        self.filename = filename
        self.model = model
        self._obj_val = None

    @ staticmethod
    def read_file(filepath: str) -> MPSFile:
        """Create an MPSFile from a filepath."""
        filename = Path(filepath).name
        model = None
        with contextlib.redirect_stdout(io.StringIO()):
            model = gp.read(filepath)

        return MPSFile(filename, model)

    @staticmethod
    def read_files(dir: str) -> List[MPSFile]:
        """Read a list of mps files from a directory."""
        mps_files = []
        for file in Path(dir).iterdir():
            if file.is_file():
                mps_file = MPSFile.read_file(str(file))
                mps_files.append(mps_file)
        return mps_files

    @staticmethod
    def write_files(dir: str, mps_files: List[MPSFile]):
        """Write a list of mps files to a directory."""
        for mps_file in mps_files:
            output_path = Path(dir) / mps_file.filename
            mps_file.model.write(str(output_path))

    def copy(self) -> MPSFile:
        """Copy an MPSFile object"""
        self.model.update()
        return MPSFile(self.filename, self.model.copy())

    def append_to_filename(self, to_append: str):
        """Append a string to the filename."""
        stem_and_extension = self.filename.split('.', 1)
        stem = stem_and_extension[0]
        extension = stem_and_extension[1]
        self.filename = stem + to_append + '.' + extension

    def optimize(self):
        """Optimize the program and save the objective value."""
        # TODO: do we need to check that self.model.Status == gp.GRB.OPTIMAL?
        self.model.setParam('OutputFlag', 0)
        self.model.update()
        self.model.optimize()
        self._obj_val = self.model.ObjVal

    def obj_val(self):
        """Lazy compute the objective value."""
        if self._obj_val == None:
            self.optimize()
        return self._obj_val

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

        input_obj = input_file.obj_val()
        output_obj = output_file.obj_val()
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

        input_obj = input_file.obj_val()
        output_obj = output_file.obj_val()
        relation = self.scale * input_obj == output_obj
        relation_str = f"{self.scale} * {input_obj} == {output_obj}"

        return relation, relation_str
