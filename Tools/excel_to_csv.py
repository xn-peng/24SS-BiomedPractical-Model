import pandas as pd

file_path = './TestSet_demographics.xlsx'
df = pd.read_excel(file_path)

dict_list = []

for index, row in df.iterrows():
    record = {
        "RandID": row['RandID'],
        "Age": row['Age'],
        "Sex": row['Sex'],
        "TSI": row['TSI'],
        "ScanManufacturer": row['ScanManufacturer'],
        "Lesion": None 
    }
    dict_list.append(record)


dict_df = pd.DataFrame(dict_list)


output_csv_path = './TestSet_demographics_with_lesion.csv'


dict_df.to_csv(output_csv_path, index=False)