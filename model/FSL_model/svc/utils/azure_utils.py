from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import pandas as pd
import json
import subprocess
import os
import ssl


def authenticate_azure():
    subscription = json.loads(subprocess.check_output('az account list --query "[?isDefault]"', shell=True).decode('utf-8'))
    ml_client = MLClient(
        DefaultAzureCredential(), subscription[0]['id'], 'COMPL_ML', "ml_compliance"
        )

    allowSelfSignedHttps(True)
    return ml_client

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context


def get_latest_training_data():
    ml_client = authenticate_azure()


    # data_asset = ml_client.data._get_latest_version(name="Training-Data-Compliance")

    ## OR

    data_asset = ml_client.data.get(name="training_data", version = 1)

    path = {
        'file': data_asset.path
    }

    import mltable#
    from mltable import MLTableHeaders, MLTableFileEncoding, DataType

    tbl = mltable.from_delimited_files(paths = [path], delimiter = ',', support_multi_line=True)

    df = tbl.to_pandas_dataframe()

    return df