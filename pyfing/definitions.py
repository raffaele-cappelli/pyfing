import numpy as np
import json
from typing import NamedTuple

# type alias for fingerprint (and other) images
Image = np.ndarray


class Minutia(NamedTuple):
    """A minutia with its (x,y) coordinates, direction, type ("E"=Ending, "B"=Bifurcation, "O"=Other), and quality"""
    x: int
    y: int
    direction: float
    type: str
    quality: float


class Parameters:
    """
    Base class for algorithm parameters.
    """
    
    def save(self, path: str):
        with open(path, 'w') as file:
            json.dump(self.__dict__, file)

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as file:
            return cls(**json.load(file))

    def __repr__(self):
        d = {k:round(v,2) if isinstance(v,float) else v for k,v in self.__dict__.items()}
        return str(d)
