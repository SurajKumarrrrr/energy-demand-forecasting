from datetime import date

import pytest

from energy_forecast.calendar import easter_sunday, german_holidays


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2022, date(2022, 4, 17)), (2023, date(2023, 4, 9)), (2024, date(2024, 3, 31)),
     (2025, date(2025, 4, 20))],
)
def test_easter_sunday(year, expected):
    assert easter_sunday(year) == expected


def test_german_holidays_2024():
    hol = german_holidays(2024)
    assert len(hol) == 9
    assert hol[date(2024, 3, 29)] == "Karfreitag"
    assert hol[date(2024, 5, 9)] == "Christi Himmelfahrt"
    assert hol[date(2024, 5, 20)] == "Pfingstmontag"
    assert date(2024, 10, 3) in hol
