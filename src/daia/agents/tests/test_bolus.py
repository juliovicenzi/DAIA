import math

import pytest

from daia.agents.tools import calculate_correction_bolus, calculate_meal_bolus
from daia.daia_db.db_model import FoodNutrientsPortionSize, Meal


@pytest.fixture
def bolus_values():
    meal = Meal(
        foods=[
            FoodNutrientsPortionSize(
                food_name="test_food",
                portion_size_g=100,
                carbohydrates_g=10,
                energy_kcal=10,
                lipid_g=10,
                protein_g=10,
                fiber_g=1,
            ),
            FoodNutrientsPortionSize(
                food_name="test_food2",
                portion_size_g=50,
                carbohydrates_g=10,
                energy_kcal=10,
                lipid_g=10,
                protein_g=10,
                fiber_g=1,
            ),
        ]
    )
    sample = {
        "meal": meal,
        "insulin_to_carb_ratio": 15.0,
        "expected_bolus": math.floor((10.0 + 5.0) / 15.0),
    }
    return sample


@pytest.fixture
def correction_values():
    sample = {
        "current_blood_glucose": 200.0,
        "target_blood_glucose": 120.0,
        "insulin_sensitivity": 45.0,
        "expected_correction": math.floor((200.0 - 120.0) / 45.0),
    }
    return sample


def test_meal_bolus(bolus_values: dict):
    result = calculate_meal_bolus.invoke(
        input={
            "meal": bolus_values["meal"],
            "insulin_to_carb_ratio": bolus_values["insulin_to_carb_ratio"],
        },
    )
    assert result == bolus_values["expected_bolus"]


def test_correction_bolus(correction_values: dict):
    result = calculate_correction_bolus.invoke(
        input={
            "current_blood_glucose": correction_values["current_blood_glucose"],
            "target_blood_glucose": correction_values["target_blood_glucose"],
            "insulin_sensitivity": correction_values["insulin_sensitivity"],
        },
    )
    assert result == correction_values["expected_correction"]


def test_zero_sensitivity(correction_values: dict):
    with pytest.raises(ValueError):
        calculate_correction_bolus.invoke(
            input={
                "current_blood_glucose": correction_values["current_blood_glucose"],
                "target_blood_glucose": correction_values["target_blood_glucose"],
                "insulin_sensitivity": 0.0,
            },
        )


def test_zero_insulin_ratio(bolus_values: dict):
    with pytest.raises(ValueError):
        calculate_meal_bolus.invoke(
            input={
                "meal": bolus_values["meal"],
                "insulin_to_carb_ratio": bolus_values["insulin_to_carb_ratio"],
                "insulin_to_carb_ratio": 0.0,
            },
        )
