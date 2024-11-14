import mlflow
from tensorboard.backend.event_processing import event_accumulator
import json
import glob
import os
import re

def parse_best_train_log(train_log):
    epoch_pattern = r"epoch\s+:\s+(\d+).*?loss\s+:\s+([\d.]+).*?loss\s+:\s+([\d.]+).*?val_loss\s+:\s+([\d.]+).*?val_accuracy\s+:\s+([\d.]+).*?val_precision\s+:\s+([\d.]+).*?val_recall\s+:\s+([\d.]+).*?val_f1_score\s+:\s+([\d.]+)"

    # Extract all matches in the form of (epoch, loss, accuracy, val_loss, val_accuracy, val_precision, val_recall, val_f1_score)
    matches = re.findall(epoch_pattern, train_log, re.DOTALL)

    # Initialize variables to keep track of the minimum validation loss and corresponding metrics
    min_val_loss = float("inf")
    best_epoch_metrics = None

    # Iterate over all matches to find the epoch with the lowest val_loss
    for match in matches:
        epoch, loss, accuracy, val_loss, val_accuracy, val_precision, val_recall, val_f1_score = match
        val_loss = float(val_loss)

        # Update the best metrics if the current val_loss is the lowest so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_epoch_metrics = {
                "epoch": int(epoch),
                "loss": float(loss),
                "accuracy": float(accuracy),
                "val_loss": val_loss,
                "val_accuracy": float(val_accuracy),
                "val_precision": float(val_precision),
                "val_recall": float(val_recall),
                "val_f1_score": float(val_f1_score),
            }
    return best_epoch_metrics


def get_experiments(base_path, exp_pattern):
    # TODO: rename variables, because they are misleading.
    #       Mlflow Experiments are a group of Runs, and
    #       we are disregarding this difference in this script.
    exps = []
    old = []
    new = []
    register = mlflow.search_runs() # dataframe with existing runs

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
            new_run = register.loc[lambda df: df["tags.mlflow.runName"]==exp_name].empty

            if new_run:
                new.append(exp_name)
                exps.append(paths)
            else:
                old.append(exp_name)

    print(f"Found {len(new)} new experiments.")
    for exp in new:
        print(f"\t{exp}")
    print(f"With {len(old)} repeated experiments (already registered in mlflow):")
    for exp in old:
        print(f"\t{exp}")
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
            # TODO: register test metrics
            with open(test_log_path) as f:
                test_log = f.read()

            pattern = r"'loss': ([\d.]+), 'accuracy': ([\d.]+)"
            match = re.search(pattern, test_log)
            if match:
                loss = float(match.group(1))
                acc = float(match.group(2))
                mlflow.log_metric('loss/test', loss)
                mlflow.log_metric('acc/test', acc)
            else:
                print("No match in test.log found.")

            # TODO: best_valid metrics
            with open(train_log_path) as f:
                train_log = f.read()

            best_valid = parse_best_train_log(train_log)
            mlflow.log_metric('loss/valid/best', loss)
            mlflow.log_metric('acc/valid/best', acc)

            # load config.json
            with open(config_path) as f:
                config = json.load(f)

            if mlflow.active_run() is not None:
                    mlflow.end_run()

            with mlflow.start_run(run_name=f"{experiment}/{run}"):
                mlflow.log_artifact(train_log_path, "log")
                mlflow.log_artifact(test_log_path, "log")
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

                # load train/valid metrics from tensorboard
                ea = event_accumulator.EventAccumulator(event_path)
                ea.Reload()
                for tag in ea.Tags()['scalars']:
                    print("\ttag:", tag)
                    events = ea.Scalars(tag)
                    for e in events:
                        # Log each metric from TensorBoard to MLflowea
                        mlflow.log_metric(tag, e.value, step=e.step)
                mlflow.end_run()

        except Exception as e:
            print(e)
