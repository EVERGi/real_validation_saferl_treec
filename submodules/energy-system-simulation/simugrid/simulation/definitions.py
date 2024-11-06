from enum import Enum


class EnergyVector(Enum):
    ELECTRIC = "electric"
    HEAT = "heat"
    COLD = "cold"
    NG = "ng" # natural gas
    HYDROGEN = "hydrogen"

    def __str__(self):
        return self.value
