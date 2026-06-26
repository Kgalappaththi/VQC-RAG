import pandas as pd
from pathlib import Path

def load_documents(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def clean_text(text: str) -> str:
    return " ".join(str(text).split())

def preprocess_documents(input_path: str, output_path: str) -> None:
    df = load_documents(input_path)
    df["text"] = df["text"].apply(clean_text)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_documents(
        "data/raw/documents.csv",
        "data/processed/documents_clean.csv"
    )