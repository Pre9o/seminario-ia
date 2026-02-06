import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scaled_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_scaled{ext}"


def scale_pair(path_a: str, path_b: str, target_column: str = "CKD progression") -> None:
    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    common_cols = cols_a & cols_b
    feature_cols = [c for c in common_cols if c != target_column]

    for col in feature_cols:
        nunique_combined = pd.concat([df_a[col], df_b[col]], axis=0).nunique(dropna=True)
        if nunique_combined <= 5:
            continue

        a_vals = df_a[col].astype(float).to_numpy()
        b_vals = df_b[col].astype(float).to_numpy()

        scaler = StandardScaler()
        combined = np.concatenate([a_vals, b_vals]).reshape(-1, 1)
        scaler.fit(combined)

        df_a[col] = scaler.transform(a_vals.reshape(-1, 1)).ravel()
        df_b[col] = scaler.transform(b_vals.reshape(-1, 1)).ravel()

    df_a.to_csv(scaled_output_path(path_a), index=False)
    df_b.to_csv(scaled_output_path(path_b), index=False)


def main() -> None:
    pairs = [
        ("datasets/dataset_filled_boruta_age_adults.csv", "datasets/dataset_filled_boruta_age_elderly.csv"),
        ("datasets/dataset_filled_boruta_etiology12.csv", "datasets/dataset_filled_boruta_etiology34.csv"),
        ("datasets/dataset_filled_boruta_stage234.csv", "datasets/dataset_filled_boruta_stage5.csv"),
    ]
    for a, b in pairs:
        scale_pair(a, b)


if __name__ == "__main__":
    main()