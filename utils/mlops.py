import mlflow
from tensorboard.backend.event_processing import event_accumulator
import json
import glob
import os

def get_experiments(base_path, exp_pattern):
    exps = []
    old = []
    new = []
    test_paths = glob.glob(f"{base_path}/log/{exp_pattern}/*/test.log")
    for test_log_path in test_paths:
        experiment, run = test_log_path.split("/")[-3:-1]
        event_path = glob.glob(f"{base_path}/log/{experiment}/{run}/events*")
        train_log_path = f"{base_path}/log/{experiment}/{run}/info.log"
        config_path = f"{base_path}/models/{experiment}/{run}/config.json"
        #model_best = f"{base_path}/models/{experiment}/{run}/model_best.pth"
        if len(event_path) == 0:
            continue
        event_path = event_path[0]
        paths = [event_path, train_log_path, test_log_path, config_path] #, model_best]

        if all(os.path.exists(path) for path in paths):
            exp_name = f"{experiment}/{run}"
            already_registered = mlflow.get_experiment_by_name(exp_name)
            if already_registered:
                old.append(exp_name)
            else:
                new.append(exp_name)
                exps.append(paths)

    print(f"Found {len(new)} new experiments.")
    for exp in new:
        print("\t{exp}")
    print(f"With {len(old)} repeated experiments (already registered in mlflow):")
    for exp in old:
        print("\t{exp}")
    print()

    return exps

if __name__ == "__main__":
    base_path = "/root/saved"
    exp_pattern = "*"
    exps = get_experiments(base_path, exp_pattern)

    mlflow.set_tracking_uri("http://autograd.live")

    for event_path, train_log_path, test_log_path, config_path in exps:
        experiment, run = config_path.split('/')[-3:-1]

        try:
            mlflow.log_artifact(train_log_path, "log")
            mlflow.log_artifact(test_log_path, "log")
            # load config.json
            with open(config_path) as f:
                config = json.load(f)

            with mlflow.start_run(run_name=f"{experiment}/{run}"):
                # load config file
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
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()
                for tag in ea.Tags()['scalars']:
                    print("\ttag:", tag)
                    events = ea.Scalars(tag)
                    for e in events:
                        # Log each metric from TensorBoard to MLflowea
                        mlflow.log_metric(tag, e.value, step=e.step)


        except Exception as e:
            print(e)
