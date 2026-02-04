import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

def fill_missing_values(dataset_path, target_column):
    file_name = dataset_path.split('.')[0].split('/')[-1]
    file_location = '/'.join(dataset_path.split('/')[:-1])
    file_extension = dataset_path.split('.')[1]
    if file_extension != 'csv':
        raise ValueError("Input file must be a CSV file.")
    # 1. Lê o arquivo
    dataset = pd.read_csv(dataset_path)
    # 2. Verifica colunas com valores ausentes e conta quantos faltam em cada uma
    missing_values_dict = dataset.isnull().sum()
    missing_values_dict = {col: int(count) for col, count in missing_values_dict.items() if count > 0}
    print("Missing values per column:", missing_values_dict)

    # # 3. Identifica colunas com menos de 2% de valores ausentes
    # total_rows = len(dataset)
    # print("Total rows in dataset:", total_rows)
    # cols_less_2pct_missing = [col for col, count in missing_values_dict.items() if count / total_rows < 0.02]
    # print("Columns with less than 2% missing values:", cols_less_2pct_missing)
    # # 4. Remove registros com valores ausentes nessas colunas
    # if cols_less_2pct_missing:
    #     dataset = dataset.dropna(subset=cols_less_2pct_missing)

    # 5. processed_dataset recebe o dataset atualizado
    continuous_columns = []
    categorical_columns = []

    for column in missing_values_dict.keys():
        if dataset[column].nunique() < 15:
            categorical_columns.append(column)
        else:
            continuous_columns.append(column)

    print("Continuous columns with missing values:", continuous_columns)
    print("Categorical columns with missing values:", categorical_columns)
            
    for column in continuous_columns:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        train_data = dataset[dataset[column].notnull()]
        test_data = dataset[dataset[column].isnull()]

        X_train = train_data.drop(columns=[column, target_column])
        y_train = train_data[column]

        X_test = test_data.drop(columns=[column, target_column])

        model.fit(X_train, y_train)
        predicted_values = model.predict(X_test)

        dataset.loc[dataset[column].isnull(), column] = predicted_values

    for column in categorical_columns:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        train_data = dataset[dataset[column].notnull()]
        test_data = dataset[dataset[column].isnull()]

        X_train = train_data.drop(columns=[column, target_column])
        y_train = train_data[column]

        X_test = test_data.drop(columns=[column, target_column])

        model.fit(X_train, y_train)
        predicted_values = model.predict(X_test)

        dataset.loc[dataset[column].isnull(), column] = predicted_values

    output_path = f"{file_location}/{file_name}_filled_rf.csv"
    dataset.to_csv(output_path, index=False)


if __name__ == "__main__":
    fill_missing_values('datasets/dataset.csv', 'CKD progression')