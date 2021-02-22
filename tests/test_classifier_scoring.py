import pytest
from scoring.classifier_scoring import (
    AggCase,
    Case,
    CellGrid,
    DataCell,
    DataLabels,
    SpatialDim,
    WeekNumber,
    _check_coordinates,
)


def test_week_numbers():
    week = WeekNumber("2020", 52)
    assert week == "2020W52"

    week = WeekNumber(2020, "14")
    assert week == "2020W14"

    week = WeekNumber("2019", "30")
    assert week == "2019W30"

    week = WeekNumber(2019, 36)
    assert week == "2019W36"

    week = WeekNumber(2019, "05")
    assert week == "2019W05"

    with pytest.raises(AssertionError):
        week = WeekNumber(2019, 58)
    with pytest.raises(AssertionError):
        week = WeekNumber(2019, "59")
    with pytest.raises(AssertionError):
        week = WeekNumber(2019, "-1")

    assert WeekNumber(2020, 5) < WeekNumber(2020, 50)
    assert WeekNumber(2020, "05") >= WeekNumber(2020, 5)
    assert WeekNumber(2020, "05") == WeekNumber("2020", 5)


def test_spatial_dim():
    # pylint: disable=unused-variable
    spatial_dim = SpatialDim(4)
    with pytest.raises(AssertionError):
        spatial_dim = SpatialDim("4")
    with pytest.raises(AssertionError):
        spatial_dim = SpatialDim(-5)

    assert SpatialDim(99) > SpatialDim(50)
    assert SpatialDim(22) == SpatialDim(22)


def test_check_coordinates():
    with pytest.raises(AssertionError):
        _ = _check_coordinates((WeekNumber(2020, 19), 5))
    with pytest.raises(AssertionError):
        _ = _check_coordinates(("2020W05", SpatialDim(5)))
    with pytest.raises(AssertionError):
        _ = _check_coordinates((SpatialDim(5), WeekNumber(2020, 19)))
    _ = _check_coordinates((WeekNumber(2020, 19), SpatialDim(5)))


def test_case_comparisons():
    case_1 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 5), SpatialDim(5)),
    )
    case_2 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 5), SpatialDim(5)),
    )
    assert case_1 == case_2

    case_3 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 5), SpatialDim(5)),
    )
    case_4 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
    )
    assert case_3 != case_4


def test_case_proba_sum():
    """Tests that a Case cannot have DataLabels with the sum of all their probas != 1"""
    with pytest.raises(AssertionError):
        _ = Case(
            data_label_probas={
                DataLabels.ONE: 0.3,
                DataLabels.TWO: 0.2,
                DataLabels.THREE: 0.2,
                DataLabels.ENDEMIC: 0.2,
                DataLabels.NON_CASE: 0.2,
            },
            coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
        )
    with pytest.raises(AssertionError):
        _ = Case(
            data_label_probas={
                DataLabels.ONE: 0,
                DataLabels.TWO: 0.2,
                DataLabels.THREE: 0.2,
                DataLabels.ENDEMIC: 0.2,
                DataLabels.NON_CASE: 0.2,
            },
            coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
        )


def test_case_completeness():
    """Tests that a Case cannot have only a subset of all DataLabels assigned"""
    with pytest.raises(AssertionError):
        _ = Case(
            data_label_probas={
                DataLabels.ENDEMIC: 0.5,
                DataLabels.NON_CASE: 0.5,
            },
            coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
        )


def test_agg_case():
    """Tests that a AggCase has completness property of Case but different DataLabel handeling"""
    with pytest.raises(AssertionError):
        # pylint: disable=unused-variable
        _ = AggCase(
            data_label_probas={
                DataLabels.ENDEMIC: 2,
                DataLabels.NON_CASE: 3,
            },
            coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
        )
        _ = AggCase(
            data_label_probas={
                DataLabels.ONE: 0,
                DataLabels.TWO: 0,
                DataLabels.THREE: 0,
                DataLabels.ENDEMIC: 0,
                DataLabels.NON_CASE: 0,
            },
            coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
        )
    _ = AggCase(
        data_label_probas={
            DataLabels.ONE: 3,
            DataLabels.TWO: 2,
            DataLabels.THREE: 2,
            DataLabels.ENDEMIC: 2,
            DataLabels.NON_CASE: 2,
        },
        coordinates=(WeekNumber(2020, 6), SpatialDim(5)),
    )


def test_data_cell_proba():
    cell = DataCell(
        [
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.4,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            AggCase(
                data_label_probas={
                    DataLabels.ONE: 3,
                    DataLabels.TWO: 2,
                    DataLabels.THREE: 1,
                    DataLabels.ENDEMIC: 1,
                    DataLabels.NON_CASE: 1,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
        ],
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    assert cell.data_label_probas == {
        DataLabels.ONE: 0.36,
        DataLabels.TWO: 0.24,
        DataLabels.THREE: 0.14,
        DataLabels.ENDEMIC: 0.12,
        DataLabels.NON_CASE: 0.14,
    }


def test_data_cell_coordinate_conformity():
    with pytest.raises(AssertionError):
        _ = DataCell(
            [
                Case(
                    data_label_probas={
                        DataLabels.ONE: 0.2,
                        DataLabels.TWO: 0.2,
                        DataLabels.THREE: 0.2,
                        DataLabels.ENDEMIC: 0.2,
                        DataLabels.NON_CASE: 0.2,
                    },
                    coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
                ),
                Case(
                    data_label_probas={
                        DataLabels.ONE: 0.4,
                        DataLabels.TWO: 0.2,
                        DataLabels.THREE: 0.2,
                        DataLabels.ENDEMIC: 0,
                        DataLabels.NON_CASE: 0.2,
                    },
                    coordinates=(WeekNumber(2020, 26), SpatialDim(5)),
                ),
            ],
            coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
        )


def test_data_cell_comparison():
    cell_1 = DataCell(
        [
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.4,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
        ],
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    cell_2 = DataCell(
        [
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.4,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
        ],
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    cell_3 = DataCell(
        [
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.3,
                    DataLabels.TWO: 0.3,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
        ],
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    cell_4 = DataCell(
        [
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.4,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 24), SpatialDim(5)),
            ),
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 24), SpatialDim(5)),
            ),
        ],
        coordinates=(WeekNumber(2020, 24), SpatialDim(5)),
    )
    assert cell_1 == cell_2
    assert cell_1 != cell_3
    assert cell_1 != cell_4


def test_data_cell_coordinates():
    _ = DataCell(
        Case(
            data_label_probas={
                DataLabels.ONE: 0.2,
                DataLabels.TWO: 0.2,
                DataLabels.THREE: 0.2,
                DataLabels.ENDEMIC: 0.2,
                DataLabels.NON_CASE: 0.2,
            },
            coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
        ),
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    with pytest.raises(AssertionError):
        _ = DataCell(
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=("2020W25", SpatialDim(5)),
            ),
            coordinates=("2020W25", SpatialDim(5)),
        )
    with pytest.raises(AssertionError):
        _ = DataCell(
            Case(
                data_label_probas={
                    DataLabels.ONE: 0.2,
                    DataLabels.TWO: 0.2,
                    DataLabels.THREE: 0.2,
                    DataLabels.ENDEMIC: 0.2,
                    DataLabels.NON_CASE: 0.2,
                },
                coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
            ),
            coordinates=(WeekNumber(2020, 25), 5),  # Here is the error
        )


def test_data_cell_case_count():
    case_1 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    case_2 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    case_3 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    cases = [case_1, case_2, case_3]
    cell = DataCell(cases, coordinates=(WeekNumber(2020, 25), SpatialDim(5)))
    assert cell.case_number(DataLabels.ONE) == 0.9
    assert cell.case_number(DataLabels.NON_CASE) == 0.3


def test_cell_grid_ranges():
    case_1 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 24), SpatialDim(4)),
    )
    case_2 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 25), SpatialDim(5)),
    )
    case_3 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 26), SpatialDim(6)),
    )
    cells = [
        DataCell(case_1, coordinates=(WeekNumber(2020, 24), SpatialDim(4))),
        DataCell(case_2, coordinates=(WeekNumber(2020, 25), SpatialDim(5))),
        DataCell(case_3, coordinates=(WeekNumber(2020, 26), SpatialDim(6))),
    ]
    grid = CellGrid(
        cells,
        time_range=(WeekNumber(2020, 24), WeekNumber(2020, 26)),
        spatial_range=(SpatialDim(4), SpatialDim(6)),
    )
    assert grid.time_range == [
        WeekNumber(2020, 24),
        WeekNumber(2020, 25),
        WeekNumber(2020, 26),
    ]
    assert all(map(lambda x: isinstance(x, WeekNumber), grid.time_range))

    assert grid.spatial_range == [
        SpatialDim(4),
        SpatialDim(5),
        SpatialDim(6),
    ]
    assert all(map(lambda x: isinstance(x, SpatialDim), grid.spatial_range))


def test_cell_grid_padding():
    case_1 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 24), SpatialDim(4)),
    )
    case_2 = Case(
        data_label_probas={
            DataLabels.ONE: 0.3,
            DataLabels.TWO: 0.2,
            DataLabels.THREE: 0.2,
            DataLabels.ENDEMIC: 0.2,
            DataLabels.NON_CASE: 0.1,
        },
        coordinates=(WeekNumber(2020, 24), SpatialDim(5)),
    )
    cells = [
        DataCell(case_1, coordinates=(WeekNumber(2020, 24), SpatialDim(4))),
        DataCell(case_2, coordinates=(WeekNumber(2020, 24), SpatialDim(5))),
    ]
    grid = CellGrid(
        cells,
        time_range=(WeekNumber(2020, 24), WeekNumber(2020, 24)),
        spatial_range=(SpatialDim(4), SpatialDim(6)),
    )

    assert grid.cells == [
        DataCell(case_1, coordinates=(WeekNumber(2020, 24), SpatialDim(4))),
        DataCell(case_2, coordinates=(WeekNumber(2020, 24), SpatialDim(5))),
        DataCell(
            Case(
                data_label_probas={
                    DataLabels.ONE: 0,
                    DataLabels.TWO: 0,
                    DataLabels.THREE: 0,
                    DataLabels.ENDEMIC: 0,
                    DataLabels.NON_CASE: 1,
                },
                coordinates=(WeekNumber(2020, 24), SpatialDim(6)),
            ),
            coordinates=(WeekNumber(2020, 24), SpatialDim(6)),
        ),
    ]
