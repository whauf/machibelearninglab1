
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


Placement_data = pd.read_csv("Placement_Data_Full_Class.csv")


def preprocess_placement_data(Placement_data):
    change_col = [
        "gender",
        "ssc_b",
        "hsc_b",
        "hsc_s",
        "degree_t",
        "workex",
        "specialisation",
        "status",
    ]

    Placement_data[change_col] = Placement_data[change_col].astype("category")

    category_list_place = list(Placement_data.select_dtypes("category"))
    Placement_hot_coded = pd.get_dummies(
        Placement_data, columns=category_list_place, drop_first=True
    )

    Placement_hot_coded = Placement_hot_coded.dropna(subset=["salary"])

    num_cols = ["ssc_p", "hsc_p", "degree_p", "etest_p", "mba_p"]
    Placement_hot_coded[num_cols] = MinMaxScaler().fit_transform(
        Placement_hot_coded[num_cols]
    )

    Placement_hot_coded = Placement_hot_coded.drop(
        ["sl_no", "status_Placed"], axis=1, errors="ignore"
    )

    Placement_hot_coded["salary_target"] = (
        Placement_hot_coded["salary"] > Placement_hot_coded["salary"].median()
    ).astype(int)

    prevplace = (
        Placement_hot_coded["salary_target"].value_counts()[1]
        / len(Placement_hot_coded["salary_target"])
    )

    Trainplace, Temppalce = train_test_split(
        Placement_hot_coded,
        train_size=0.50,
        stratify=Placement_hot_coded["salary_target"],
    )

    Tuneplace, Testplace = train_test_split(
        Temppalce, test_size=0.50, stratify=Temppalce["salary_target"]
    )

    return Trainplace, Tuneplace, Testplace, prevplace


Trainplace, Tuneplace, Testplace, prevplace = preprocess_placement_data(Placement_data)
print(Trainplace.shape, Tuneplace.shape, Testplace.shape)
print(prevplace)
print(
    Trainplace["salary_target"].mean(),
    Tuneplace["salary_target"].mean(),
    Testplace["salary_target"].mean(),
)


CC_data = pd.read_csv("cc_institution_details.csv")


def preprocess_CC_data(CC_data):
    change_col2 = [
        "level",
        "control",
        "basic",
        "hbcu",
        "flagship",
        "state",
        "city",
        "site",
        "similar",
        "nicknames",
    ]

    CC_data[change_col2] = CC_data[change_col2].astype("category")

    state_region = {
        "ME": "Northeast",
        "NH": "Northeast",
        "VT": "Northeast",
        "MA": "Northeast",
        "RI": "Northeast",
        "CT": "Northeast",
        "NY": "Northeast",
        "NJ": "Northeast",
        "PA": "Northeast",

        "DE": "South",
        "MD": "South",
        "DC": "South",
        "VA": "South",
        "WV": "South",
        "NC": "South",
        "SC": "South",
        "GA": "South",
        "FL": "South",
        "KY": "South",
        "TN": "South",
        "AL": "South",
        "MS": "South",
        "AR": "South",
        "LA": "South",
        "OK": "South",
        "TX": "South",

        "OH": "Midwest",
        "IN": "Midwest",
        "MI": "Midwest",
        "WI": "Midwest",
        "IL": "Midwest",
        "MN": "Midwest",
        "IA": "Midwest",
        "MO": "Midwest",
        "ND": "Midwest",
        "SD": "Midwest",
        "NE": "Midwest",
        "KS": "Midwest",

        "MT": "West",
        "WY": "West",
        "CO": "West",
        "NM": "West",
        "AZ": "West",
        "UT": "West",
        "NV": "West",
        "ID": "West",
        "WA": "West",
        "OR": "West",
        "CA": "West",
        "AK": "West",
        "HI": "West",
    }

    CC_data["region"] = CC_data["state"].map(state_region)
    CC_data["region"] = CC_data["region"].astype("category")

    category_list_cc = list(CC_data.select_dtypes("category"))

    CC_data_hot_coded = pd.get_dummies(
        CC_data, columns=category_list_cc, drop_first=True
    )

    target = "grad_150_value"

    num_cols_cc = (
        CC_data_hot_coded.select_dtypes(include=["int64", "float64"]).columns.drop(target)
    )

    CC_data_hot_coded[num_cols_cc] = MinMaxScaler().fit_transform(
        CC_data_hot_coded[num_cols_cc]
    )

    CC_data_hot_coded[num_cols_cc].describe()

    CC_data_hot_coded = CC_data_hot_coded.drop(
        ["index", "unitid", "chronname", "city", "site", "similar", "nicknames"],
        axis=1,
        errors="ignore",
    )

    CC_data_hot_coded["grad_target"] = (
        CC_data_hot_coded["grad_150_value"] > CC_data_hot_coded["grad_150_value"].median()
    ).astype(int)

    prevCC = (
        CC_data_hot_coded["grad_target"].value_counts()[1]
        / len(CC_data_hot_coded["grad_target"])
    )

    cc_dt = CC_data_hot_coded.dropna(subset=["grad_target"])

    Train, Temp = train_test_split(
        cc_dt, train_size=0.55, random_state=42, stratify=cc_dt["grad_target"]
    )

    Tune, Test = train_test_split(
        Temp, test_size=0.50, random_state=42, stratify=Temp["grad_target"]
    )

    return Train, Tune, Test, prevCC


Train, Tune, Test, prevCC = preprocess_CC_data(CC_data)
print(Train.shape, Tune.shape, Test.shape)
print(prevCC)
print(
    Train["grad_target"].mean(), Tune["grad_target"].mean(), Test["grad_target"].mean()
)
