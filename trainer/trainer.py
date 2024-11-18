import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker, plot_confusion_matrix
from torchvision.utils import make_grid


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device, data_loader,
                 writer_step='epoch', valid_data_loader=None, lr_scheduler=None, len_epoch=None, register_img_batch=False):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))
        self.register_img_batch = register_img_batch
        self.img_step = max(self.log_step, data_loader.batch_size//2)

        self.writer_step = writer_step # 'batch' or 'epoch'

        # Every MetricTracker Update is registered using writer
        metric_writer = self.writer if self.writer_step=='batch' else None

        self.train_metrics = MetricTracker('loss', 'lr', *[m.__name__ for m in self.metric_ftns], writer=metric_writer)
        self.valid_metrics = MetricTracker('loss', 'lr', *[m.__name__ for m in self.metric_ftns], writer=metric_writer)
        self.train_metrics.init_confusion_matrix(self.model.num_classes)
        self.valid_metrics.init_confusion_matrix(self.model.num_classes)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, (data, target) in enumerate(self.data_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            if self.lr_scheduler is not None:
                if type(self.lr_scheduler).__name__ in ["CyclicLR"]:
                    self.lr_scheduler.step()

            if self.writer_step == 'batch':
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            else:
                self.writer.set_step(epoch-1)

            self.train_metrics.update('loss', loss.item())

            if self.lr_scheduler is not None:
                self.train_metrics.update('lr', self.lr_scheduler.get_last_lr()[0])

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            self.train_metrics.update_confusion_matrix(output, target)

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if self.register_img_batch and (self.writer_step=='batch') and (batch_idx % self.img_step == 0):
                self.writer.add_image('input', make_grid(data.cpu()[:16], nrow=8, normalize=True))

            if batch_idx == self.len_epoch:
                break
    
        self.train_metrics.get_other_metrics()
        log = self.train_metrics.result()

        if self.register_img_batch and (self.writer_step=='epoch'):
            self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

        if self.writer_step == 'epoch':
            for key, val in log.items():
                if 'confusion_matrix' in key or 'recall_' in key or 'precision_' in key:
                    continue
                self.writer.add_scalar(key, val)
            self.writer.add_image("Train Confusion Matrix", plot_confusion_matrix(self.train_metrics.confusion_matrix))

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.writer_step == 'epoch':
            for key, val in val_log.items():
                if 'confusion_matrix' in key or 'recall_' in key or 'precision_' in key:
                    continue
                self.writer.add_scalar(key, val)
            self.writer.add_image("Valid Confusion Matrix", plot_confusion_matrix(self.valid_metrics.confusion_matrix, title="Valid Confusion Matrix"))

        if self.lr_scheduler is not None:
            if type(self.lr_scheduler).__name__ in ["ReduceLROnPlateau"]:
                self.lr_scheduler.step(val_log['loss'])
            elif type(self.lr_scheduler).__name__ in ["CyclicLR"]:
                pass
            else:
                self.lr_scheduler.step()

        log["train_confusion_matrix"] = "\n" + str(self.train_metrics.confusion_matrix)
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                if self.writer_step == 'batch':
                    self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                else:
                    self.writer.set_step(epoch-1, 'valid')

                self.valid_metrics.update('loss', loss.item())

                self.valid_metrics.update_confusion_matrix(output, target)
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

            self.valid_metrics.get_other_metrics()

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')

        log = self.valid_metrics.result()
        log["valid_confusion_matrix"] = "\n" + str(self.valid_metrics.confusion_matrix)
        return log

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
