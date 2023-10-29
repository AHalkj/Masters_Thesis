import numpy as np
from datetime import datetime
import pandas as pd
import mltable#
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import uuid
from svc.utils import authenticate_azure


def get_old_data():
    ml_client = authenticate_azure()

    print(ml_client)

    # data_asset = ml_client.data._get_latest_version(name="Training-Data-Compliance")

    ## OR

    data_asset = ml_client.data.get(name="training_data", version = 1)

    path = {
        'file': data_asset.path
    }

    tbl = mltable.from_delimited_files(paths = [path], delimiter = ',', support_multi_line=True)
    df = tbl.to_pandas_dataframe().drop(columns = ['Column1'])
    return df


def join_dataframes_by_block(filepath):
    df_list = []

    import os
    span_count = 0
    for folder in[x[0] for x in os.walk(filepath)][1:]:
        for file in os.listdir(folder):
           if file.endswith('.csv'):
                file_name = folder + '/' + file

                df = pd.read_csv(file_name, index_col= 0)[['block_num', 'text', 'label']]
                span_count = span_count + len(df)
                df = df.groupby('block_num').agg({'text': lambda x:' '.join(str(x)), 'label': lambda x: 1*(np.mean(x) > 0)}).reset_index().drop(columns = ['block_num']).assign(
                    likelihood = -1,
                    encryptionID = '000',
                    dateAdded = datetime.today()
                )
                df_list.append(df)

    big_df = pd.concat(df_list)
    return big_df

def upload_to_azure(filepath):
    ml_client = authenticate_azure()

    my_data = Data(
        path=filepath,
        type=AssetTypes.URI_FILE,
        description="Uploaded through Python SDK. First time",
        name="training_data", 
        # the version gets automatically updated to most recent when using this
    )

    ml_client.data.create_or_update(my_data)

if __name__ == '__main__':
    file_path = 'C:/Users/KLWE/OneDrive - SimCorp/AHAKLWE/data/labelled'
    df = join_dataframes_by_block(file_path)
    df_old = get_old_data()
    df = pd.concat([df_old, df])
    

    # filepath gets equipped with uuid, so that we don't create a data asset form a fle that we didn't just create with the same code . 
    filepath = f'training_data_{uuid.uuid4().hex}.csv'
    df.to_csv(filepath)
    upload_to_azure(filepath)
