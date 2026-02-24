import os
import pandas as pd
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def adults_and_elderly(df):
    return df[df["age"] >= 60], "age_elderly", df[df["age"] < 60], "age_adults"


def etiology_groups(df):
    return df[df["etiology of CKD"] < 3], "etiology_12", df[df["etiology of CKD"] >= 3], "etiology_34"


def stage_groups(df):
    return df[df["CKD_stage"] < 5], "stage_234", df[df["CKD_stage"] >= 5], "stage_5"


def scale_features(train_df, test_df, val_df):
    continuous_cols = [f for f in train_df.columns if train_df[f].nunique() > 13]

    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])

    train_cont_scaled = pd.DataFrame(scaler.transform(train_df[continuous_cols]), columns=continuous_cols, index=train_df.index)
    test_cont_scaled = pd.DataFrame(scaler.transform(test_df[continuous_cols]), columns=continuous_cols, index=test_df.index)
    val_cont_scaled = pd.DataFrame(scaler.transform(val_df[continuous_cols]), columns=continuous_cols, index=val_df.index)

    train_final = pd.concat([train_df.drop(columns=continuous_cols), train_cont_scaled], axis=1)
    test_final = pd.concat([test_df.drop(columns=continuous_cols), test_cont_scaled], axis=1)
    val_final = pd.concat([val_df.drop(columns=continuous_cols), val_cont_scaled], axis=1)

    return train_final, test_final, val_final


def fill_missing_values(train_df, df_to_fill):
    train_filled = train_df.copy()
    df_filled = df_to_fill.copy()
    
    cols_with_missing = []
    for col in train_filled.columns:
        n_missing = train_filled[col].isna().sum() + df_filled[col].isna().sum()
        if n_missing > 0:
            cols_with_missing.append((col, n_missing))
    
    cols_with_missing.sort(key=lambda x: x[1])
    cols_order = [col for col, _ in cols_with_missing]
    complete_cols = [col for col in train_filled.columns if col not in cols_order]
    
    for col in cols_order:
        is_categorical = train_df[col].nunique() < 13
        feature_cols = complete_cols.copy()
        
        train_complete_mask = train_filled[col].notna()
        for feat_col in feature_cols:
            train_complete_mask &= train_filled[feat_col].notna()
        
        if train_complete_mask.sum() < 10:
            if is_categorical:
                fill_value = train_df[col].mode(dropna=True)[0]
            else:
                fill_value = train_df[col].mean()
            train_filled[col] = train_filled[col].fillna(fill_value)
            df_filled[col] = df_filled[col].fillna(fill_value)
        else:
            X_train = train_filled.loc[train_complete_mask, feature_cols]
            y_train = train_filled.loc[train_complete_mask, col]
            
            if is_categorical:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X_train, y_train)
            
            train_missing_mask = train_filled[col].isna()
            if train_missing_mask.any():
                X_pred_train = train_filled.loc[train_missing_mask, feature_cols]
                train_filled.loc[train_missing_mask, col] = model.predict(X_pred_train)
            
            fill_missing_mask = df_filled[col].isna()
            if fill_missing_mask.any():
                X_pred_fill = df_filled.loc[fill_missing_mask, feature_cols]
                df_filled.loc[fill_missing_mask, col] = model.predict(X_pred_fill)
        
        complete_cols.append(col)
    
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
                        target_progression, split_feature=None):
    features_to_remove = ['ID', 'gender', 'eGFR(0M)', 'eGFR(6M)', 'eGFR(12M)', 'eGFR(18M)', 'eGFR(24M)', 'eGFR(30M)', 'eGFR(36M)', '50%eGFR_reached(6M)',
                          '50%eGFR_reached(12M)', '50%eGFR_reached(18M)', '50%eGFR_reached(24M)', '50%eGFR_reached(30M)', '50%eGFR_reached(36M)',
                          'eGFR(last visit)', '50%eGFR_reached', '50%eGFR_duration', 'CKD category', 'UPCR category', 'dip-stick proteinuria',
                          'Cr', '50%eGFR1_reached(18M)', '50%eGFR', 'CKD_stage', 'RRT', 'RRT_duration', 'death', 'death_duration', 'CKD progression_duration', 'development of CVD_duration']

    df_a = df_a.drop(columns=features_to_remove, errors='ignore')
    df_b = df_b.drop(columns=features_to_remove, errors='ignore')

    df_a = df_a.dropna(subset=[target_progression])
    X_a = df_a.drop(columns=[target_progression])
    progression_a = df_a[target_progression]

    X_a_train, X_a_test_and_val, progression_a_train, progression_a_test_and_val = train_test_split(
        X_a,
        progression_a,
        test_size=0.3,
        random_state=42,
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

    forest = RandomForestClassifier(n_jobs=-1, max_depth=5)
    boruta_selector = BorutaPy(estimator=forest, n_estimators="auto", random_state=42)
    boruta_selector.fit(unified_X_train.values, unified_target.values)

    selected_features = unified_X_train.columns[boruta_selector.support_].tolist()
    selected_features.remove(split_feature) if split_feature in selected_features else None
    selected_features = selected_features[:10]
    
    # if 'dip-stick proteinuria' in selected_features:
    #     features_to_remove.append('proteinuria')
    # elif 'proteinuria' in selected_features:
    #     features_to_remove.append('dip-stick proteinuria')
    # if 'CKD_stage' in selected_features:
    #     features_to_remove.append('CKD category')
    # elif 'CKD category' in selected_features:
    #     features_to_remove.append('CKD_stage')
    # if 'UPCR' in selected_features:
    #     features_to_remove.append('UPCR category')
    # elif 'UPCR category' in selected_features:
    #     features_to_remove.append('UPCR')

    feature_ranking = sorted(
        zip(unified_X_train.columns.tolist(), boruta_selector.ranking_.tolist()),
        key=lambda x: x[1],
    )
    print("Feature ranking:", feature_ranking)   

    if split_feature and split_feature in X_a_train.columns:
        X_a_train = X_a_train.drop(columns=[split_feature])
        X_a_test = X_a_test.drop(columns=[split_feature])
        X_a_val = X_a_val.drop(columns=[split_feature])

    if split_feature and split_feature in X_b_train.columns:
        X_b_train = X_b_train.drop(columns=[split_feature])
        X_b_test = X_b_test.drop(columns=[split_feature])
        X_b_val = X_b_val.drop(columns=[split_feature])

    selected_features = [f for f in selected_features if f not in features_to_remove]
    print("Selected features:", selected_features)

    X_a_train_selected = X_a_train[selected_features]
    X_a_test_selected = X_a_test[selected_features]
    X_a_val_selected = X_a_val[selected_features]

    X_b_train_selected = X_b_train[selected_features]
    X_b_test_selected = X_b_test[selected_features]
    X_b_val_selected = X_b_val[selected_features]

    X_a_train, X_a_test, X_a_val = scale_features(X_a_train_selected, X_a_test_selected, X_a_val_selected)
    X_b_train, X_b_test, X_b_val = scale_features(X_b_train_selected, X_b_test_selected, X_b_val_selected)

    save_split_dataset(
        output_folder,
        name_a,
        X_a_train, progression_a_train,
        X_a_val, progression_a_val,
        X_a_test, progression_a_test
    )

    save_split_dataset(
        output_folder,
        name_b,
        X_b_train, progression_b_train,
        X_b_val, progression_b_val,
        X_b_test, progression_b_test
    )


def main():
    ds = pd.read_csv("datasets/original/dataset_original.csv")
    target_progression = "CKD progression"

    output_base = "datasets_processed_2"
    os.makedirs(output_base, exist_ok=True)

    df_elderly, name_elderly, df_adults, name_adults = adults_and_elderly(ds)
    age_folder = os.path.join(output_base, "age")
    os.makedirs(age_folder, exist_ok=True)
    pre_process_dataset(
        df_elderly, name_elderly,
        df_adults, name_adults,
        age_folder,
        target_progression,
        split_feature="age"
    )

    df_etiology_12, name_12, df_etiology_34, name_34 = etiology_groups(ds)
    etiology_folder = os.path.join(output_base, "etiology")
    os.makedirs(etiology_folder, exist_ok=True)
    pre_process_dataset(
        df_etiology_12, name_12,
        df_etiology_34, name_34,
        etiology_folder,
        target_progression,
        split_feature="etiology of CKD"
    )

    df_stage_234, name_234, df_stage_5, name_5 = stage_groups(ds)
    stage_folder = os.path.join(output_base, "stage")
    os.makedirs(stage_folder, exist_ok=True)
    pre_process_dataset(
        df_stage_234, name_234,
        df_stage_5, name_5,
        stage_folder,
        target_progression,
        split_feature="CKD_stage"
    )


if __name__ == "__main__":
    main()
