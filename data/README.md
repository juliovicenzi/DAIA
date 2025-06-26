# Data sources for DAIA

Our data source is the [FoodData Central](https://fdc.nal.usda.gov/download-datasets)

The CSV files required to run our [cleaning script](../src/daia/daia_db/data_cleaning.ipynb) can be downloaded through this [FNDDS link](https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip).

The process for cleaning the data is documented in the [cleaning script](../src/daia/daia_db/data_cleaning.ipynb).
The process of creating embeddings and writing it to a PGSQL database is documented in the [embedding script](../src/daia/daia_db/write_db.py).

Both of these are required to setup a database with data for DAIA's agents to function.
