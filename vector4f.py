from dataclasses import dataclass, field
import numpy as np

@dataclass
class Vector4f:
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 1
    idx: int = 0

    def __init__(self, x=0, y=0, z=0, idx=0):
        self.x = x
        self.y = y
        self.z = z
        self.idx = idx

    def to_array(self):
        return  np.array([self.x, self.y, self.z, self.w])

    def normilize(self):
        length = np.sqrt(self.x**2 + self.y**2 + self.z**2)
        self.x /= length
        self.y /= length
        self.z /= length
        self.w = 1
        return self

