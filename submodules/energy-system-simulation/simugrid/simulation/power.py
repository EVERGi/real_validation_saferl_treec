from __future__ import annotations


class Power:
    """
    :ivar _electrical: the electrical power
    :type _electrical: float
    :ivar _heating: the heating power
    :type _heating: float
    """

    def __init__(self, electrical: float = 0.0, heating: float = 0.0):
        """

        :param electrical: the electrical power
        :param heating: the heating power
        """

        self._electrical = electrical
        self._heating = heating

    @property
    def total_power(self) -> float:
        return self._electrical + self._heating

    @property
    def electrical(self) -> float:
        return self._electrical

    @property
    def heating(self) -> float:
        return self._heating

    def set_new_power(self, electrical: float = 0.0, heating: float = 0.0):
        self._electrical = electrical
        self._heating = heating

    def __sub__(self, other: Power) -> Power:
        return Power(self._electrical - other.electrical, self._heating - other.heating)

    def __isub__(self, other: Power) -> Power:
        self._electrical -= other.electrical
        self._heating -= other.heating
        return self

    def __add__(self, other: Power) -> Power:
        return Power(self._electrical + other.electrical, self._heating + other.heating)

    def __iadd__(self, other: Power) -> Power:
        self._electrical += other.electrical
        self._heating += other.heating
        return self

    def __neg__(self) -> Power:
        return Power(-self._electrical, -self._heating)

    def __eq__(self, other: Power) -> bool:
        return self._electrical == other.electrical and self._heating == other.heating

    def __ne__(self, other: Power) -> bool:
        return self._electrical != other.electrical and self._heating != other.heating

    def __lt__(self, other: Power) -> tuple[bool, bool]:
        # <
        return self._electrical < other.electrical, self._heating < self.heating

    def __gt__(self, other: Power) -> tuple[bool, bool]:
        # >
        return self._electrical > other.electrical, self._heating > self.heating

    def __le__(self, other: Power) -> tuple[bool, bool]:
        # <=
        return self._electrical <= other.electrical, self._heating <= self.heating

    def __ge__(self, other: Power) -> tuple[bool, bool]:
        # >=
        return self._electrical >= other.electrical, self._heating >= self.heating

    def __repr__(self):
        return f"(electrical power={self._electrical}, heating power={self._heating})"

    def transfer(self, other: Power) -> Power:
        # self >> other
        # producer >> demand
        electrical_transfer = 0.0
        heating_transfer = 0.0

        if other.electrical < 0 and self._electrical > 0:
            electrical_transfer = min(self._electrical, -other.electrical)
        elif other._electrical > 0 and self._electrical < 0:
            electrical_transfer = max(self._electrical, -other.electrical)

        if other.heating < 0 and self._heating > 0:
            heating_transfer = min(self._heating, -other.heating)
        elif other._heating > 0 and self._heating < 0:
            heating_transfer = max(self._heating, -other.heating)

        return Power(electrical_transfer, heating_transfer)

    def __abs__(self) -> Power:
        return Power(abs(self._electrical), abs(self._heating))

    @property
    def flow(self):
        return self._electrical != 0 or self._heating != 0

    @property
    def empty(self) -> tuple[bool, bool]:
        return self._electrical == 0, self._heating == 0

    @property
    def vectorised(self) -> list[float]:
        return [self._electrical, self._heating]
