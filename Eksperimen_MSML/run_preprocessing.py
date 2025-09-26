import os
import pandas as pd
from preprocessing.automate_MuhammadAbiyaMakruf import preprocess_data


if __name__ == "__main__":
    raw_path = "insurance_raw.csv"
    save_pipeline_path = "preprocessing/preprocessor.joblib"
    save_header_path = "preprocessing/insurance_preprocessing/columns.csv"
    save_dataset_path = "preprocessing/insurance_preprocessing"

    os.makedirs(save_dataset_path, exist_ok=True)

    # Load data
    df = pd.read_csv(raw_path)

    # Jalankan preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(
        data=df,
        target_column="charges",
        save_path=save_pipeline_path,
        file_path=save_header_path
    )

    # Simpan hasil preprocessing ke CSV
    pd.DataFrame(X_train).assign(target=y_train.reset_index(drop=True)) \
        .to_csv(f"{save_dataset_path}/insurance_train_preprocessed.csv", index=False)

    pd.DataFrame(X_test).assign(target=y_test.reset_index(drop=True)) \
        .to_csv(f"{save_dataset_path}/insurance_test_preprocessed.csv", index=False)