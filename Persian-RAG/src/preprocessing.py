
import re
from tqdm.auto import tqdm

from src.config import MIN_CHARS


def normalize_persian_text(text):

    if not isinstance(text, str):
        return ""

    # Arabic Yeh variants
    text = text.replace("ي", "ی")
    text = text.replace("ى", "ی")

    # Arabic Kaf
    text = text.replace("ك", "ک")

    # Remove BOM / zero-width space
    text = text.replace("\ufeff", "")
    text = text.replace("\u200b", "")

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def filter_documents(df):

    print("Initial documents:", len(df))

    # Main Wikipedia namespace only
    df = df[df["Namespace"] == 0].copy()

    # Remove disambiguation pages
    df = df[
        ~df["IsDisambiguationPage"]
    ].copy()

    # Remove missing title/text
    df = df[
        df["Title"].notna() &
        df["Text"].notna()
    ].copy()

    print("After structural filtering:", len(df))

    return df


def clean_documents(df):

    tqdm.pandas()

    df["clean_title"] = df["Title"].progress_apply(
        normalize_persian_text
    )

    df["clean_text"] = df["Text"].progress_apply(
        normalize_persian_text
    )

    # Remove short documents
    df = df[
        df["clean_text"].str.len() >= MIN_CHARS
    ].copy()

    print("After length filtering:", len(df))

    # Remove duplicate documents
    before = len(df)

    df = df.drop_duplicates(
        subset=["clean_text"]
    ).copy()

    print("Removed duplicates:", before - len(df))

    print("Final documents:", len(df))

    return df


def preprocess_documents(df):

    df = filter_documents(df)

    df = clean_documents(df)

    return df
