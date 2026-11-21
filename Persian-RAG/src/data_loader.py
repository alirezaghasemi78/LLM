
from datasets import load_dataset, Dataset, load_from_disk
import pandas as pd

from src.config import (
    DATA_DIR,
    DATASET_NAME,
    DATASET_SPLIT,
    MAX_DOCUMENTS
)


def load_wikipedia_dataset():

    # Check whether dataset already exists locally
    if DATA_DIR.exists():

        print("Loading dataset from local disk...")

        dataset = load_from_disk(str(DATA_DIR))

    else:

        print("Streaming dataset from Hugging Face...")

        streamed_dataset = load_dataset(
            DATASET_NAME,
            split=DATASET_SPLIT,
            streaming=True
        )

        # Get only the first N records
        rows = list(streamed_dataset.take(MAX_DOCUMENTS))

        print(f"Received {len(rows)} records.")

        print("Saving dataset locally...")

        dataset = Dataset.from_list(rows)

        DATA_DIR.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        dataset.save_to_disk(str(DATA_DIR))

        print("Dataset saved successfully!")

    return dataset


def load_wikipedia_dataframe():

    dataset = load_wikipedia_dataset()

    df = dataset.to_pandas()

    print("\nDataset loaded successfully!")
    print("Number of samples:", len(df))

    return df
