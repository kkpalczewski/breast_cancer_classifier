import pandas as pd
import numpy as np
import subprocess
import os

def check_single_accuracy(data_folder, metadata_path, path_to_results):
    metadata = pd.read_csv(metadata_path, index_col=0)
    new_columns = ['benign_4a', 'benign_4b', 'malignant_4a', 'malignant_4b', 'final_prediction']
    for column in new_columns:
        metadata[column] = np.nan

    for idx, record in metadata.iterrows():
        file_path = record['image file path']
        view = record['image view']
        left_or_right_breast = record['left or right breast']

        if left_or_right_breast == "RIGHT":
            left_or_right_breast = "R"
        elif left_or_right_breast == "LEFT":
            left_or_right_breast = "L"

        separator = "-"
        description = separator.join([left_or_right_breast, view])
        cmd = ['bash', '/home/krzysztof/Documents/Studia/Master_thesis/05_Baseline_model_NYU/run_single.sh', os.path.join(data_folder, file_path), description]
        rc = subprocess.run(cmd, stdout=subprocess.PIPE)
        print("Prediction made on file: {}".format(file_path))
        result_4a = eval(rc.stdout.decode('utf-8').split("\n")[-4])
        result_4b = eval(rc.stdout.decode('utf-8').split("\n")[-2])

        metadata.at[idx, 'benign_4a'] = result_4a['benign']
        metadata.at[idx, 'malignant_4a'] = result_4a['malignant']
        metadata.at[idx, 'benign_4b'] = result_4b['benign']
        metadata.at[idx, 'malignant_4b'] = result_4b['malignant']

        if result_4b['malignant'] >= result_4b['benign']:
            metadata.at[idx, 'final_prediction'] = 1
        else:
            metadata.at[idx, 'final_prediction'] = 0

    try:
        metadata.to_csv(path_to_results)
        print("[SUCCESS] Predictions saved to: {}".format(path_to_results))
    except Exception as e:
        print("[FAILURE] Unable to save metadata in: {}, due to error: {}".format(path_to_results, e))



if __name__=="__main__":
    os.chdir("/home/krzysztof/Documents/Studia/Master_thesis/05_Baseline_model_NYU")
    metadata_path = "/home/krzysztof/Documents/Studia/Master_thesis/02_Source_code/breast-cancer-research/data/labels/full_metadata_without_master_folder.csv"
    data_folder = "/media/krzysztof/ADATA_HD700/Breast_cancer_PNG/CBIS-DDSM"

    path_to_results = "./prediction.csv"
    check_single_accuracy(data_folder, metadata_path, path_to_results)
