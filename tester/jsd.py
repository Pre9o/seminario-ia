import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon


def _is_continuous_numeric(series, min_unique_values=20):
    return pd.api.types.is_numeric_dtype(series) and series.dropna().nunique() > min_unique_values


def _categorical_distribution(series):
    normalized = series.astype("object").where(series.notna(), "__NaN__")
    return normalized.value_counts(normalize=True)


def _numeric_distribution(series, bin_edges):
    binned = pd.cut(series, bins=bin_edges, include_lowest=True, duplicates="drop")
    normalized = binned.astype("object").where(binned.notna(), "__NaN__")
    return normalized.value_counts(normalize=True)

def load_dataset(source_dataset, target_dataset):
    train_source = pd.read_csv(f'{source_dataset}/train.csv')
    train_target = pd.read_csv(f'{target_dataset}/train.csv')

    test_source = pd.read_csv(f'{source_dataset}/test.csv')
    test_target = pd.read_csv(f'{target_dataset}/test.csv')

    val_source = pd.read_csv(f'{source_dataset}/val.csv')
    val_target = pd.read_csv(f'{target_dataset}/val.csv')

    return pd.concat([train_source, test_source, val_source]), pd.concat([train_target, test_target, val_target])


def compute_jsd_between_datasets(source_dataset, target_dataset):
    source_data, target_data = load_dataset(source_dataset, target_dataset)

    jsd_values = {}
    for column in source_data.columns:
        if column in target_data.columns:
            source_column = source_data[column]
            target_column = target_data[column]

            if _is_continuous_numeric(source_column) and _is_continuous_numeric(target_column):
                combined_non_null = pd.concat([source_column, target_column], ignore_index=True).dropna()

                if combined_non_null.empty:
                    jsd_values[column] = 0.0
                    continue

                unique_values = combined_non_null.nunique()
                number_of_bins = min(30, max(5, int(np.sqrt(len(combined_non_null)))))
                number_of_bins = min(number_of_bins, int(unique_values))

                if number_of_bins < 2:
                    source_dist = _categorical_distribution(source_column)
                    target_dist = _categorical_distribution(target_column)
                else:
                    bin_edges = np.histogram_bin_edges(combined_non_null.to_numpy(), bins=number_of_bins)
                    source_dist = _numeric_distribution(source_column, bin_edges)
                    target_dist = _numeric_distribution(target_column, bin_edges)
            else:
                source_dist = _categorical_distribution(source_column)
                target_dist = _categorical_distribution(target_column)

            all_categories = sorted(set(source_dist.index).union(set(target_dist.index)), key=str)
            source_dist = source_dist.reindex(all_categories, fill_value=0.0)
            target_dist = target_dist.reindex(all_categories, fill_value=0.0)

            js_distance = jensenshannon(source_dist, target_dist, base=2)
            js_divergence = float(js_distance ** 2)
            jsd_values[column] = js_divergence

    return jsd_values

def main():
    source_dataset = 'datasets_processed_2/stage/stage_234'
    target_dataset = 'datasets_processed_2/stage/stage_5'

    source_df, target_df = load_dataset(source_dataset, target_dataset)
    jsd_values = compute_jsd_between_datasets(source_dataset, target_dataset)

    print("Jensen-Shannon Divergence between source and target datasets:")
    for column, jsd in jsd_values.items():
        print(f"{column}: {jsd:.4f}")
        
if __name__ == "__main__":
    main()
