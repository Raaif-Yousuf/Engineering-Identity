import numpy as np
import pandas as pd

from ei_model.experiments.cleaning import MAX_HOURS_PER_WEEK, clean_implausible_hours


def test_nulls_values_over_the_physical_bound():
    df = pd.DataFrame(
        {
            "Hours on Campus Per Week": [10.0, 300.0, 123456789.0, 168.0, 169.0],
            "unrelated": [1, 2, 3, 4, 5],
        }
    )

    cleaned, report = clean_implausible_hours(df)

    assert report == {"Hours on Campus Per Week": 3}
    assert cleaned["unrelated"].tolist() == [1, 2, 3, 4, 5]
    assert cleaned["Hours on Campus Per Week"].iloc[0] == 10.0
    assert cleaned["Hours on Campus Per Week"].iloc[3] == MAX_HOURS_PER_WEEK
    assert np.isnan(cleaned["Hours on Campus Per Week"].iloc[1])
    assert np.isnan(cleaned["Hours on Campus Per Week"].iloc[2])
    assert np.isnan(cleaned["Hours on Campus Per Week"].iloc[4])


def test_missing_columns_are_ignored():
    df = pd.DataFrame({"some_other_column": [1, 2, 3]})

    cleaned, report = clean_implausible_hours(df)

    assert report == {}
    pd.testing.assert_frame_equal(cleaned, df)


def test_does_not_mutate_input():
    df = pd.DataFrame({"Total Per Week": [10.0, 999.0]})
    original = df.copy()

    clean_implausible_hours(df)

    pd.testing.assert_frame_equal(df, original)
