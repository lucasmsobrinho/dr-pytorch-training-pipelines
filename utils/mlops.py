from tensorboard.backend.event_processing import event_accumulator
import mlflow
import glob
import argparse

def register_tensorboard():
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        for e in events:
            # Log each metric from TensorBoard to MLflow
            mlflow.log_metric(tag, e.value, step=e.step)



if __name__ == "__main__":
    args = argparse.ArgumentParser(description='PyTorch Template')
    
    events = 
    # Path to TensorBoard log files
    tensorboard_log_dir = "/path/to/tensorboard/logs"

    # Load TensorBoard event data
    ea = event_accumulator.EventAccumulator(tensorboard_log_dir)
    ea.Reload()

    # Start an MLflow run
    with mlflow.start_run():
        # Loop through scalar metrics and log them to MLflow
    



