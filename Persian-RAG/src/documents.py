
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np


TYPE_MAP = {
    0: "Unknown",
    1: "Person",
    2: "Location",
    3: "Organization",
    4: "Event",
    6: "Other"
}


@dataclass
class Document:

    page_content: str
    metadata: Dict[str, Any]


def convert_to_python(value):
    """
    Convert NumPy values into standard Python types
    so they can be serialized as JSON.
    """

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            key: convert_to_python(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            convert_to_python(item)
            for item in value
        ]

    return value


def row_to_document(row):

    title = row["clean_title"]
    text = row["clean_text"]

    content = (
        f"عنوان: {title}\n\n"
        f"{text}"
    )

    metadata = {
        "doc_id": int(row["Id"]),
        "title": title,
        "type": int(row["Type"]),
        "type_name": TYPE_MAP.get(
            int(row["Type"]),
            "Unknown"
        ),
        "rank": int(row["Rank"]),
        "categories": convert_to_python(row["Parents"]),
    }

    return Document(
        page_content=content,
        metadata=metadata
    )


def create_documents(df):

    documents = []

    for row in df.to_dict(orient="records"):

        documents.append(
            row_to_document(row)
        )

    return documents
