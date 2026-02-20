import os
import pandas as pd
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier




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


def pre_process_dataset(df, name, output_folder, target_progression):
    X = df.drop(columns=[target_progression])
    progression = df[target_progression]
    X_train, X_test_and_val, progression_train, progression_test_and_val = train_test_split(
        X,
        progression,
        test_size=0.3,
        random_state=42,
        stratify=progression
    )

    X_test, X_val, progression_test, progression_val = train_test_split(
        X_test_and_val,
        progression_test_and_val,
        test_size=0.5,
        random_state=42,
        stratify=progression_test_and_val
    )

    # Scaleing
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    cat_columns = []
    continuous_columns = []
    for col in X_train.columns:
        if X_train[col].nunique() < 13:
            cat_columns.append(col)
        else:
            continuous_columns.append(col)

    X_train_cat = X_train[cat_columns]
    X_val_cat = X_val[cat_columns]
    X_test_cat = X_test[cat_columns]

    X_train_cont = scaler.fit_transform(X_train[continuous_columns])
    X_val_cont = scaler.transform(X_val[continuous_columns])
    X_test_cont = scaler.transform(X_test[continuous_columns])
    
    X_train = pd.concat(
        [X_train_cat, pd.DataFrame(X_train_cont, columns=continuous_columns, index=X_train.index)],
        axis=1,
    )
    X_val = pd.concat(
        [X_val_cat, pd.DataFrame(X_val_cont, columns=continuous_columns, index=X_val.index)],
        axis=1,
    )
    X_test = pd.concat(
        [X_test_cat, pd.DataFrame(X_test_cont, columns=continuous_columns, index=X_test.index)],
        axis=1,
    )

    save_split_dataset(
        output_folder,
        name,
        X_train, progression_train,
        X_val, progression_val,
        X_test, progression_test
    )


def main():
    df_adults = pd.read_csv("datasets/dataset_filled_boruta_age_adults.csv")
    df_elderly = pd.read_csv("datasets/dataset_filled_boruta_age_elderly.csv")

    df_etiology_12 = pd.read_csv("datasets/dataset_filled_boruta_etiology12.csv")
    df_etiology_34 = pd.read_csv("datasets/dataset_filled_boruta_etiology34.csv")

    df_stage_234 = pd.read_csv("datasets/dataset_filled_boruta_stage234.csv")
    df_stage_5 = pd.read_csv("datasets/dataset_filled_boruta_stage5.csv")

    target_progression = "CKD progression"

    output_base = "datasets_processed_3"
    os.makedirs(output_base, exist_ok=True)

    name_elderly = 'age_elderly'
    name_adults = 'age_adults'

    age_folder = os.path.join(output_base, "age")
    os.makedirs(age_folder, exist_ok=True)

    pre_process_dataset(
        df_elderly, name_elderly,
        age_folder,
        target_progression,
    )
    pre_process_dataset(
        df_adults, name_adults,
        age_folder,
        target_progression,
    )

    name_etiology_12 = 'etiology_12'
    name_etiology_34 = 'etiology_34'

    etiology_folder = os.path.join(output_base, "etiology")
    os.makedirs(etiology_folder, exist_ok=True)
    
    pre_process_dataset(
        df_etiology_12, name_etiology_12,
        etiology_folder,
        target_progression,
    )

    pre_process_dataset(
        df_etiology_34, name_etiology_34,
        etiology_folder,
        target_progression,
    )

    name_stage_234 = 'stage_234'
    name_stage_5 = 'stage_5'

    stage_folder = os.path.join(output_base, "stage")
    os.makedirs(stage_folder, exist_ok=True)

    stage_folder = os.path.join(output_base, "stage")
    os.makedirs(stage_folder, exist_ok=True)

    pre_process_dataset(
        df_stage_234, name_stage_234,
        stage_folder,
        target_progression,
    )
    pre_process_dataset(
        df_stage_5, name_stage_5,
        stage_folder,
        target_progression,
    )

if __name__ == "__main__":
    main()
