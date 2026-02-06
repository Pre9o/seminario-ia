import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from collections import namedtuple
scaler = StandardScaler()

RANDOM_STATE=42
TEST_SIZE=0.2

class Dataset:
    def __init__(self, file_name, target_column=None):
        # load file to dataset
        self.ds = pd.read_csv(file_name)
        self.target_column = target_column

        # for col_index in range(self.ds.shape[1]):
        #     col_name = self.ds.columns[col_index]
            
        #     if target_column and col_name == target_column:
        #         continue
            
        #     unique_values = self.ds.iloc[:, col_index].nunique()
        #     if unique_values <= 5:
        #         # categorical feature - não normaliza
        #         continue
        #     else:
        #         self.ds[col_name] = self.ds[col_name].astype(float)
        #         scaled_values = scaler.fit_transform(self.ds[col_name].values.reshape(-1, 1))
        #         self.ds[col_name] = scaled_values.flatten()  

        if target_column is not None:
            # split features(x) and target(y)
            self.features = self.ds.drop(target_column, axis=1)
            self.target = self.ds[target_column]
            # split into train and test sets
            # by default with test_size=0.2
            self.split(0.6, 0.2, 0.2)

    def setTargetColumn(self, target_column):
        self.target_column = target_column  
        # split features(x) and target(y)
        self.features = self.ds.drop(target_column, axis=1)
        self.target = self.ds[target_column]
        # split into train and test sets
        # by default with test_size=0.2
        self.split(0.8, 0.1, 0.1)

    def __repr__(self):
        return (f"Dataset(shape={self.get_shape()})")

    def split(self, train_size, validation_size, test_size, random_state=RANDOM_STATE):
        features_train_val, self.features_test, target_train_val, self.target_test = train_test_split(
            self.features,
            self.target,
            test_size=test_size,
            random_state=random_state,
            stratify=self.target
        )

        # Segundo split: separa treino de validação
        # Calcula a proporção de validação em relação ao conjunto treino+validação
        val_size_adjusted = validation_size / (train_size + validation_size)

        self.features_train, self.features_validation, self.target_train, self.target_validation = train_test_split(
            features_train_val,
            target_train_val,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=target_train_val
        )

    def get_shape(self):
        return self.features.shape[1]

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Error: Please provide the dataset path as the first argument.")
        sys.exit(1)
    file_name = sys.argv[1]
    if not isinstance(file_name, str) or not file_name.endswith('.csv'):
        print("Error: The dataset path must be a string ending with '.csv'.")
        sys.exit(1)
    
    dataset = Dataset(file_name, 'CKD progression')
    print(dataset)
    print(dataset.get_shape())