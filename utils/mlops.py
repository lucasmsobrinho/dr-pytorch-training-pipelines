import mlflow
from tensorboard.backend.event_processing import event_accumulator
import json
import glob

base_path ="/mnt/c/Users/tex/Downloads/saved"
events_path = f"{base_path}/saved/log/*/*/events*"
eps = glob.glob(events_path)
print(len(eps))

for ep in eps:
    print(f"run {ep}")
    experiment, run = ep.split('/')[-3:-1]
    config_path = f"{base_path}/saved/models/{experiment}/{run}/config.json"

    try:
        with open(config_path) as f:
            config = json.load(f)

        with mlflow.start_run(run_name=run):
            # loading config file for run parameters
            for main_key, value in config.items():
                if type(value) == dict:
                    for sub_key, sub_value in value.items():
                        if type(sub_value) == dict:
                            for sub_sub_key, sub_sub_value in sub_value.items():
                                # Use "main_key.sub_key" format for hierarchical logging
                                mlflow.log_param(f"{main_key}.{sub_key}.{sub_sub_key}", sub_sub_value)
                        else:
                            mlflow.log_param(f"{main_key}.{sub_key}", sub_value)
                else:
                    mlflow.log_param(main_key, value)

            # load metrics from tensorboard
            ea = event_accumulator.EventAccumulator(ep)
            ea.Reload()
            for tag in ea.Tags()['scalars']:
                print("\ttag:", tag)
                events = ea.Scalars(tag)
                for e in events:
                    # Log each metric from TensorBoard to MLflowea
                    mlflow.log_metric(tag, e.value, step=e.step)

            # load info.log
            info_path = f"{base_path}/saved/log/{experiment}/{run}/info.log"
            mlflow.log_artifact(info_path, "info.log")
            try:
                test_path = f"{base_path}/saved/log/{experiment}/{run}/test.log"
                mlflow.log_artifact(test_path, "test.log")
            except Exception as e:
                print(e)


    except Exception as e:
        print(e)

            # TODO: load train, valid, test final metrics