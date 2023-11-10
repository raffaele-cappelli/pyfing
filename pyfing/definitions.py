import numpy as np
import json

# type alias for fingerprint images
Image = np.ndarray

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
