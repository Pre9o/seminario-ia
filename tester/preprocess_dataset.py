import os
import pandas as pd
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def adults_and_elderly(df):
    return df[df["age"] >= 60], "age_elderly", df[df["age"] < 60], "age_adults"


def etiology_groups(df):
    return df[df["etiology of CKD"] < 3], "etiology_12", df[df["etiology of CKD"] >= 3], "etiology_34"


def stage_groups(df):
    return df[df["CKD_stage"] < 5], "stage_234", df[df["CKD_stage"] >= 5], "stage_5"


def fill_missing_values(train_df, df_to_fill):
    df_filled = df_to_fill.copy()

    for column in train_df.columns:
        if train_df[column].nunique() < 13:
            fill_value = train_df[column].mode(dropna=True)[0]
        else:
            fill_value = train_df[column].mean()

        df_filled[column] = df_filled[column].fillna(fill_value)

    return df_filled


def save_split_dataset(base_folder, dataset_name,
                       X_train, y_train,
                       X_val, y_val,
                       X_test, y_test):
    folder_path = os.path.join(base_folder, dataset_name)
    os.makedirs(folder_path, exist_ok=True)

    train_df = X_train.copy()
    train_df["CKD progression"] = y_train.values
    train_df.to_csv(os.path.join(folder_path, "train.csv"), index=False)

    val_df = X_val.copy()
    val_df["CKD progression"] = y_val.values
    val_df.to_csv(os.path.join(folder_path, "val.csv"), index=False)

    test_df = X_test.copy()
    test_df["CKD progression"] = y_test.values
    test_df.to_csv(os.path.join(folder_path, "test.csv"), index=False)


def pre_process_dataset(df_a, name_a, df_b, name_b, output_folder,
                        target_progression, feature_to_remove=None):
    forbidden_features = ["CKD category", "dip-stick proteinuria"]

    df_a = df_a.dropna(subset=[target_progression])
    X_a = df_a.drop(columns=[target_progression])
    progression_a = df_a[target_progression]

    X_a_train, X_a_test_and_val, progression_a_train, progression_a_test_and_val = train_test_split(
        X_a,
        progression_a,
        test_size=0.3,
        random_state=42,
        stratify=progression_a
    )

    X_a_test, X_a_val, progression_a_test, progression_a_val = train_test_split(
        X_a_test_and_val,
        progression_a_test_and_val,
        test_size=0.5,
        random_state=42,
        stratify=progression_a_test_and_val
    )

    X_a_train = fill_missing_values(X_a_train, X_a_train)
    X_a_test = fill_missing_values(X_a_train, X_a_test)
    X_a_val = fill_missing_values(X_a_train, X_a_val)

    df_b = df_b.dropna(subset=[target_progression])
    X_b = df_b.drop(columns=[target_progression])
    progression_b = df_b[target_progression]

    X_b_train, X_b_test_and_val, progression_b_train, progression_b_test_and_val = train_test_split(
        X_b,
        progression_b,
        test_size=0.3,
        random_state=42,
        stratify=progression_b
    )

    X_b_test, X_b_val, progression_b_test, progression_b_val = train_test_split(
        X_b_test_and_val,
        progression_b_test_and_val,
        test_size=0.5,
        random_state=42,
        stratify=progression_b_test_and_val
    )

    X_b_train = fill_missing_values(X_b_train, X_b_train)
    X_b_test = fill_missing_values(X_b_train, X_b_test)
    X_b_val = fill_missing_values(X_b_train, X_b_val)

    unified_X_train = pd.concat([X_a_train, X_b_train], axis=0)
    unified_target = pd.concat([progression_a_train, progression_b_train], axis=0)

    forest = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
    boruta_selector = BorutaPy(estimator=forest, n_estimators="auto", random_state=42)
    boruta_selector.fit(unified_X_train.values, unified_target.values)

    selected_features = unified_X_train.columns[boruta_selector.support_].tolist()
    selected_features = selected_features[:12]
    selected_features = [f for f in selected_features if f not in forbidden_features]
    print("Selected features:", selected_features)
    print("Feature ranking:", unified_X_train.columns[boruta_selector.ranking_].tolist())

    X_a_train_selected = X_a_train[selected_features]
    X_a_test_selected = X_a_test[selected_features]
    X_a_val_selected = X_a_val[selected_features]

    if feature_to_remove and feature_to_remove in X_a_train_selected.columns:
        X_a_train_selected = X_a_train_selected.drop(columns=[feature_to_remove])
        X_a_test_selected = X_a_test_selected.drop(columns=[feature_to_remove])
        X_a_val_selected = X_a_val_selected.drop(columns=[feature_to_remove])

    X_b_train_selected = X_b_train[selected_features]
    X_b_test_selected = X_b_test[selected_features]
    X_b_val_selected = X_b_val[selected_features]

    if feature_to_remove and feature_to_remove in X_b_train_selected.columns:
        X_b_train_selected = X_b_train_selected.drop(columns=[feature_to_remove])
        X_b_test_selected = X_b_test_selected.drop(columns=[feature_to_remove])
        X_b_val_selected = X_b_val_selected.drop(columns=[feature_to_remove])

    save_split_dataset(
        output_folder,
        name_a,
        X_a_train_selected, progression_a_train,
        X_a_val_selected, progression_a_val,
        X_a_test_selected, progression_a_test
    )

    save_split_dataset(
        output_folder,
        name_b,
        X_b_train_selected, progression_b_train,
        X_b_val_selected, progression_b_val,
        X_b_test_selected, progression_b_test
    )


def main():
    ds = pd.read_csv("datasets/original/dataset.csv")
    target_progression = "CKD progression"

    output_base = "datasets_processed"
    os.makedirs(output_base, exist_ok=True)

    df_elderly, name_elderly, df_adults, name_adults = adults_and_elderly(ds)
    age_folder = os.path.join(output_base, "age")
    os.makedirs(age_folder, exist_ok=True)
    pre_process_dataset(
        df_elderly, name_elderly,
        df_adults, name_adults,
        age_folder,
        target_progression,
        feature_to_remove="age"
    )

    df_etiology_12, name_12, df_etiology_34, name_34 = etiology_groups(ds)
    etiology_folder = os.path.join(output_base, "etiology")
    os.makedirs(etiology_folder, exist_ok=True)
    pre_process_dataset(
        df_etiology_12, name_12,
        df_etiology_34, name_34,
        etiology_folder,
        target_progression,
        feature_to_remove="etiology of CKD"
    )

    df_stage_234, name_234, df_stage_5, name_5 = stage_groups(ds)
    stage_folder = os.path.join(output_base, "stage")
    os.makedirs(stage_folder, exist_ok=True)
    pre_process_dataset(
        df_stage_234, name_234,
        df_stage_5, name_5,
        stage_folder,
        target_progression,
        feature_to_remove="CKD_stage"
    )


if __name__ == "__main__":
    main()
