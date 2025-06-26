import argparse
import math
import time
from pathlib import Path

import pandas as pd
from numpy import array_split
from sqlalchemy import Engine, text
from sqlmodel import SQLModel
from tqdm import tqdm

from daia.agents.models import get_embedding_model
from daia.daia_db.db_model import FoodNutrients, get_engine


def create_db(engine: Engine):
    SQLModel.metadata.create_all(engine)


def vectorize_index(
    df: pd.DataFrame, engine: Engine, chunk_size: int = 30, throttle_time: int = 60
):
    """Vectorize food name using OpenAI embedding model, add to the db

    Args:
        df (pd.DataFrame): input dataframe with 'food_name' column
        chunk_size (int, optional): number of rows per request. Defaults to 30.
        throttle_time (int, optional): throttle time between requests in seconds. Set to 0 to disable. Defaults to 60.
    """
    # vectorize data
    model = get_embedding_model()
    # set retry parameters for this function only
    model.retry_min_seconds = 5
    model.max_retries = 3

    # skip rows already present in db:
    df_existing = pd.read_sql(
        text(f"SELECT food_name FROM {FoodNutrients.__tablename__}"), engine
    )

    df = df[~df["food_name"].isin(df_existing["food_name"])]

    df_chunks = [
        df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)
    ]

    for df_chunk in tqdm(df_chunks):
        df_chunk.loc[:, "food_embedding"] = model.embed_documents(
            df_chunk["food_name"].tolist()
        )

        # remove rows with NaN vectors after embedding
        df_chunk = df_chunk[
            df_chunk["food_embedding"].apply(
                lambda r: not any(math.isnan(x) for x in r)
            )
        ]

        df_chunk.to_sql(
            FoodNutrients.__tablename__, engine, if_exists="append", index=False
        )

        # wait throttle_time_s seconds before next request
        time.sleep(throttle_time)


def parse_args():
    parser = argparse.ArgumentParser("DAIA DB: script to write data to db")

    def check_csv(_path: str):
        path = Path(_path)
        if not path.exists():
            raise argparse.ArgumentTypeError(f"{path} does not exist")
        if not path.suffix == ".csv":
            raise argparse.ArgumentTypeError(f"{path} is not a csv file")
        return path

    parser.add_argument(
        "--data",
        "-d",
        type=check_csv,
        required=True,
        help="Path to the data file csv file",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    engine = get_engine()
    create_db(engine)
    df = pd.read_csv(args.data)
    df = vectorize_index(df, engine)


if __name__ == "__main__":
    main()
