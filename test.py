import argparse
import torch
from tqdm import tqdm
import data_loader.ddr_dataloader as module_loader
import data_loader.ddr_dataset as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from utils.util import MetricTracker
from parse_config import ConfigParser
from utils.preprocess import get_transform, class_reduction_transform


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    preprocess = get_transform(img_size=256) # if needed can be initialized from config file

    num_classes_trained = 5
    test_set = config.init_obj('test_set', module_data, transform=preprocess)
    data_loader = config.init_obj('data_loader', module_loader, dataset=test_set)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume,  map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))
    test_metrics = MetricTracker('accuracy')

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #


            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size
            test_metrics.update_confusion_matrix(output, target)
        test_metrics.get_other_metrics()

    test_result = test_metrics.result()
    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    log.update(**{'test_'+k : v for k, v in test_result.items()})
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
