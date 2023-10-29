import sys
from azure.ai.ml import command, Input, Output
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Environment
from azure.ai.ml.sweep import Choice, Uniform, MedianStoppingPolicy


sys.path.append("C:\Repos\AHAKLWE")
from utils import authenticate_azure

def main():
    update_env = False
    ml_client = authenticate_azure()

    compute_resource = ml_client.compute.get('AHAK1')

    # data_asset = ml_client.data.get(name="training_data", version = 4)

    job_input = {"input_data": Input(type=AssetTypes.URI_FILE, path="../../dataset_creation/output/230822_training_data.csv"),
                 "learning_rate": 2e-5,
                 "undersampling_rate": 0.7,
                 "batch_size": 32}
    job_output = {"model" : Output(type=AssetTypes.MLFLOW_MODEL, mode='upload')}

    # model = Model(
    #     path= job_output["model"].path,
    #     type=AssetTypes.MLFLOW_MODEL,
    #     name="BERT_Class",
    #     description="Transformer based model build through AML V2",
    # )

    # ml_client.models.create_or_update(job_output["model"])

    # create environment
    env_name = "bert-environ"
    if update_env == True:
        job_env = Environment(
            name=env_name,
            description="Environment for training Fewshot-Model",
            conda_file="./conda.yaml", ### TODO
            # image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:latest",
            # try out this image because supposedly it trains a lot faster
            image = "AzureML-ACPT-pytorch-1.12-py39-cuda11.6-gpu:3"
        )
        job_env = ml_client.environments.create_or_update(job_env)

        print(
            f"Environment with name {job_env.name} is registered to workspace, the environment version is {job_env.version}"
        )
    else:
        job_env = ml_client.environments.list(env_name)
        env_versions = []
        for name in job_env:
            env_versions.append(int(name.version))
        job_env = ml_client.environments.get(env_name, version = max(env_versions))


    experiment_name = 'FSL'
    jobs = []
    for run in ml_client.jobs.list():
        if run.experiment_name == experiment_name:
            jobs.append(int(run.display_name.split('_')[-1]))

    if len(jobs)>0:
        job_name = experiment_name + '_' + str(max(jobs)+1)
    else:
        job_name = experiment_name + '_0'

    print(f'Submitting job {job_name}')
    # job_name = "BERT_CLASSIFICATION_45"


    # job_input = {"input_data": Input(type=AssetTypes.URI_FILE, path=data_asset_path)}
    # job_name = "BERT_CLASSIFICATION_7"
    command_job = command(
        experiment_name = experiment_name,
        compute=compute_resource.name,
        name = job_name,
        code="./src",
        command="python train_and_evaluate.py --train_path ${{inputs.input_data}} \
                                 --model_dir ${{outputs.model}} \
                                 --learning_rate ${{inputs.learning_rate}}\
                                 --undersampling_rate ${{inputs.undersampling_rate}}\
                                 --batch_size ${{inputs.batch_size}}",
        environment=f"{env_name}:{job_env.version}",
        inputs = job_input,
        outputs=job_output,
        display_name=job_name
    )

    hyperparameter = True
    if hyperparameter:
        command_job_for_sweep = command_job(
            learning_rate=Choice(values=[5e-5, 3e-5, 2e-5]),
            undersampling_rate = Uniform(min_value=0.1, max_value=1),
            batch_size = Choice(values=[16,32,64])
        )

        sweep_job = command_job_for_sweep.sweep(
        compute=compute_resource.name,
        sampling_algorithm="random",
        primary_metric="Validation Recall",
        goal="Maximize",
        )

        # Specify your experiment details
        sweep_job.display_name = "FSL_model_deployment1"
        sweep_job.experiment_name = "FSL_deployment"
        sweep_job.description = "Run and optimize FSL model for azure"

        # Define the limits for this sweep
        sweep_job.set_limits(max_total_trials=20, max_concurrent_trials=2, timeout=7200)

        # Set early stopping on this one
        sweep_job.early_termination = MedianStoppingPolicy(
            delay_evaluation=3, evaluation_interval=1
        )
        # submit the sweep
        returned_sweep_job = ml_client.create_or_update(sweep_job)
        # get a URL for the status of the job
        returned_sweep_job.services["Studio"].endpoint

    else:
        # submit the command
        returned_job = ml_client.jobs.create_or_update(command_job)
        # get a URL for the status of the job
        returned_job.studio_url

if __name__ == '__main__':
    main()



