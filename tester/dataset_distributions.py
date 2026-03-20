import pandas as pd

def analyze_dataset_distributions(train, val, test, target_column):
    combined = pd.concat([train, val, test], ignore_index=True)
    distributions = combined[target_column].value_counts(normalize=True)
    return distributions

def main():
    dataset_base_folder = 'datasets_processed_2/stage/stage_5'
    target_column = 'CKD progression'

    train = pd.read_csv(f"{dataset_base_folder}/train.csv")
    val = pd.read_csv(f"{dataset_base_folder}/val.csv")
    test = pd.read_csv(f"{dataset_base_folder}/test.csv")

    distributions = analyze_dataset_distributions(train, val, test, target_column)

    print(f"Total distributions: {distributions}")
    print(f"Total samples: {len(train) + len(val) + len(test)}")

if __name__ == "__main__":
    main()