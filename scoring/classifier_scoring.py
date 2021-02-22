from collections import Counter
from datetime import datetime
from enum import Enum, auto
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


class DataLabels(Enum):
    """An enum class to make outbreak data labels immutable"""

    ONE = auto()
    TWO = auto()
    THREE = auto()
    ENDEMIC = auto()
    NON_CASE = auto()


class WeekNumber:
    def __init__(self, year: Union[str, int], week: Union[str, int]) -> None:
        assert 53 >= int(week) >= 1, "Week must be between 1 and 53"
        self.weeknumber = f"{str(year)}W{str(week).zfill(2)}"

    def __str__(self) -> str:
        return self.weeknumber

    def __repr__(self) -> str:
        return f"WeekNumber(weeknumber='{self.weeknumber}')"

    def __eq__(self, other) -> bool:
        return self.weeknumber == other

    def __lt__(self, other) -> bool:
        return self.weeknumber < other

    def __le__(self, other) -> bool:
        return self.weeknumber <= other

    def __ne__(self, other) -> bool:
        return self.weeknumber != other

    def __ge__(self, other) -> bool:
        return self.weeknumber >= other

    def __gt__(self, other) -> bool:
        return self.weeknumber > other

    def __add__(self, other) -> str:
        return self.weeknumber + other

    def __hash__(self) -> int:
        return hash(self.weeknumber)


class SpatialDim:
    def __init__(self, spatial_id: int):
        assert isinstance(spatial_id, int), "SpatialDim must be an integer"
        assert spatial_id >= 0, "SpatialDim must positive"
        self.spatial_id = spatial_id

    def __str__(self) -> str:
        return str(self.spatial_id)

    def __repr__(self) -> str:
        return f"SpatialDim(spatial_id='{self.spatial_id}')"

    def __eq__(self, other) -> bool:
        return self.spatial_id == other

    def __lt__(self, other) -> bool:
        return self.spatial_id < other

    def __le__(self, other) -> bool:
        return self.spatial_id <= other

    def __ne__(self, other) -> bool:
        return self.spatial_id != other

    def __ge__(self, other) -> bool:
        return self.spatial_id >= other

    def __gt__(self, other) -> bool:
        return self.spatial_id > other

    def __hash__(self) -> int:
        return hash(self.spatial_id)


def _check_coordinates(
    coordinates: Tuple[WeekNumber, SpatialDim]
) -> Tuple[WeekNumber, SpatialDim]:
    assert isinstance(
        coordinates[0], WeekNumber
    ), f"The first dimension of 'coordinate' must be a WeekNumber. It is {type(coordinates[0])}"
    assert isinstance(
        coordinates[1], SpatialDim
    ), f"The second dimension of 'coordinate' must be a SpatialDim. It is {type(coordinates[0])}"
    return coordinates


class Case:
    """A case is an individual that is described by its point in space and outbreaks they belong to"""

    def __init__(
        self,
        coordinates: Tuple[WeekNumber, SpatialDim],
        data_label_probas: Optional[Dict[DataLabels, float]] = None,
    ) -> None:
        """A case always assigns a probability of having one of all available DataLables"""
        self.disease_probas = self._parse_data_label_probas(data_label_probas)
        self.coordinates = _check_coordinates(coordinates)

    def __str__(self) -> str:
        return str(self.disease_probas, self.coordinates)

    def __repr__(self) -> str:
        return f"Case(disease_probas='{self.disease_probas}', coordinates='{self.coordinates}')"

    def __eq__(self, other) -> bool:
        return (self.disease_probas == other.disease_probas) and (
            set(self.coordinates) - set(other.coordinates) == set()
        )

    def __hash__(self) -> int:
        return hash(str(self.disease_probas))

    def _parse_data_label_probas(
        self, data_label_probas: Optional[Dict[DataLabels, float]]
    ) -> Dict[DataLabels, float]:
        if data_label_probas is None:
            disease_probas = {d: 0 for d in DataLabels if d is not DataLabels.NON_CASE}
            disease_probas.update({DataLabels.NON_CASE: 1})
        else:
            disease_probas = data_label_probas

        assert (
            round(sum(disease_probas.values()), 5) == 1
        ), "The probability for all DataLables must sum up to 1"
        assert (
            set(DataLabels) - set(disease_probas.keys())
        ) == set(), "All DataLabels must be used for a Case"
        return disease_probas


class AggCase(Case):
    def __init__(
        self,
        coordinates: Tuple[WeekNumber, SpatialDim],
        data_label_probas: Optional[Dict[DataLabels, float]] = None,
    ) -> None:
        """An AggCase always assigns a case count for all available DataLables"""
        self.disease_probas = self._parse_data_label_probas(data_label_probas)
        self.coordinates = _check_coordinates(coordinates)

    def _parse_data_label_probas(
        self, data_label_probas: Optional[Dict[DataLabels, float]]
    ) -> Dict[DataLabels, float]:
        if data_label_probas is None:
            disease_probas = {d: 0 for d in DataLabels if d is not DataLabels.NON_CASE}
            disease_probas.update({DataLabels.NON_CASE: 1})
        else:
            disease_probas = data_label_probas

        assert all(
            [isinstance(x, int) for x in disease_probas.values()]
        ), "DataLabels must be assigned integrs since they represent case counts"
        assert (
            set(DataLabels) - set(disease_probas.keys())
        ) == set(), "All DataLabels must be used for a Case"
        assert (
            sum(disease_probas.values()) >= 1
        ), "The case count for all DataLables must in sum be larger than equals to 1"
        return disease_probas


class DataCell:
    "A DataCell is the space in which cases can occure"

    def __init__(
        self,
        cases: Union[List[Case], Case],
        coordinates: Tuple[WeekNumber, SpatialDim],
    ) -> None:
        if isinstance(cases, Case):
            cases = [cases]
        assert all(
            map(lambda x: isinstance(x, Case), cases)
        ), "Only cases are allowed as input"
        self.cases = cases
        self.coordinates = _check_coordinates(coordinates)
        self.data_label_probas = self._calculate_data_label_probas()
        self._check_coordinates_of_cases()

    def __str__(self) -> str:
        return str(self.data_label_probas)

    def __repr__(self) -> str:
        return f"DataCell(data_label_probas={self.data_label_probas})"

    def __eq__(self, other: "DataCell") -> bool:
        return (set(self.cases) - set(other.cases) == set()) and (
            self.coordinates == other.coordinates
        )

    def __hash__(self) -> int:
        return hash(str(self.cases) + str(self.coordinates))

    def _check_coordinates_of_cases(self):
        assert all(
            case.coordinates == self.coordinates for case in self.cases
        ), "All Cases must have the same coordinates in a DataCell"

    def _calculate_data_label_probas(self):
        unnormalized_label_probas = {
            data_label: self.case_number(data_label) for data_label in DataLabels
        }
        sum_of_probas = sum(list(unnormalized_label_probas.values()))
        return {
            k: round(v / sum_of_probas, 5) for k, v in unnormalized_label_probas.items()
        }

    def case_number(self, data_label: DataLabels) -> float:
        case_probas = [case.disease_probas[data_label] for case in self.cases]
        return round(sum(case_probas), 5)


class CellGrid:
    "A grid to align DataCells"

    def __init__(
        self,
        cells: List[DataCell],
        time_range: Tuple[WeekNumber, WeekNumber],
        spatial_range: Tuple[SpatialDim, SpatialDim],
    ):
        self.cells = cells
        self.time_range = self._create_time_range(time_range)
        self.spatial_range = self._create_spatial_range(spatial_range)
        self.cells.extend(self._pad_cells(self.cells))

    def _create_time_range(self, time_range: Tuple[WeekNumber, WeekNumber]):
        start_date = datetime.strptime(time_range[0] + "-1", "%GW%V-%u")
        end_date = datetime.strptime(time_range[1] + "-1", "%GW%V-%u")
        date_range = pd.date_range(start_date, end_date, freq="W-MON")
        return [WeekNumber(date.year, date.week) for date in date_range]

    def _create_spatial_range(self, spatial_range: Tuple[SpatialDim, SpatialDim]):
        id_range = range(spatial_range[0].spatial_id, spatial_range[1].spatial_id + 1)
        return [SpatialDim(id_) for id_ in id_range]

    def _pad_cells(self, cells):
        full_grid_coordinates = product(self.time_range, self.spatial_range)
        non_case_coordinates = set(list(full_grid_coordinates)) - set(
            [cell.coordinates for cell in cells]
        )
        return [
            DataCell(
                Case(
                    data_label_probas={
                        DataLabels.ONE: 0,
                        DataLabels.TWO: 0,
                        DataLabels.THREE: 0,
                        DataLabels.ENDEMIC: 0,
                        DataLabels.NON_CASE: 1,
                    },
                    coordinates=coordinates,
                ),
                coordinates=coordinates,
            )
            for coordinates in non_case_coordinates
        ]
