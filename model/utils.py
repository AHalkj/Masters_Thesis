import json
import subprocess
import ssl
import os
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential




def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

def authenticate_azure():
    subscription = json.loads(subprocess.check_output('az account list --query "[?isDefault]"', shell=True).decode('utf-8'))
    ml_client = MLClient(
        DefaultAzureCredential(), '231aa2de-e0fe-4a25-baab-f7e3e5a96802', 'COMPL_ML', "ml_compliance"
        )

    allowSelfSignedHttps(True)
    return ml_client