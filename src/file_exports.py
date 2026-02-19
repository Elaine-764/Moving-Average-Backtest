import os
import pandas as pd

def export_file(file_name, output_folder, data):
    output_path = os.path.join(output_folder, file_name)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(0)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    data.to_csv(output_path, index=True)
    print(f'Exported data as file {output_folder}{file_name} successfully.')
    return