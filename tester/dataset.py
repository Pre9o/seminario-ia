import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
from collections import namedtuple
scaler = StandardScaler()

RANDOM_STATE=42
TEST_SIZE=0.2

class Dataset:
    def __init__(self, folder_name, dataset_name, target_column=None):
        self.train = pd.read_csv(f"{folder_name}/{dataset_name}/train.csv")
        self.validation = pd.read_csv(f"{folder_name}/{dataset_name}/val.csv")
        self.test = pd.read_csv(f"{folder_name}/{dataset_name}/test.csv")

        self.features_train = self.train.drop(columns=[target_column])
        self.target_train = self.train[target_column]

        self.features_validation = self.validation.drop(columns=[target_column])
        self.target_validation = self.validation[target_column]

        self.features_test = self.test.drop(columns=[target_column])
        self.target_test = self.test[target_column]


    def __repr__(self):
        return (f"Dataset(shape={self.get_shape()})")

    def get_shape(self):
        return self.features_train.shape[1]

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