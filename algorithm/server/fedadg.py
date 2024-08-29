from typing import OrderedDict, Tuple
import torch

from algorithm.client.fedavg import FedAvgClient
from model.models import get_FedADG_models
from utils.optimizers_shcedulers import CosineAnnealingLRWithWarmup, get_optimizer
from utils.tools import trainable_params, get_best_device, local_time


class FedADGClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super().__init__(args, dataset, client_id, logger)
        self.E0 = self.args.num_epochs

    def initialize_model(self):
        self.classification_model, self.discriminator, self.generator = get_FedADG_models(
            self.args.model, self.args.dataset
        )
        self.device = None
        self.optimizer = get_optimizer(self.classification_model, self.args)
        self.disc_optimizer = get_optimizer(self.discriminator, self.args)
        self.gen_optimizer = get_optimizer(self.generator, self.args)

    def get_model_weights(self) -> Tuple[OrderedDict]:
        return (
            self.classification_model.state_dict(),
            self.discriminator.state_dict(),
            self.generator.state_dict(),
        )

    def load_model_weights(self, model_weights):
        self.classification_model.load_state_dict(model_weights[0])
        self.discriminator.load_state_dict(model_weights[1])
        self.generator.load_state_dict(model_weights[2])

    def move2new_device(self):
        device = get_best_device(self.args.use_cuda)
        self.classification_model.to(device)
        self.generator.to(device)
        self.discriminator.to(device)
        if self.device is None or self.device != device:
            self.device = device
            self.optimizer = get_optimizer(self.classification_model, self.args).load_state_dict(
                self.optimizer.state_dict()
            )
            self.disc_optimizer = get_optimizer(self.discriminator, self.args).load_state_dict(
                self.disc_optimizer.state_dict()
            )
            self.gen_optimizer = get_optimizer(self.generator, self.args).load_state_dict(
                self.gen_optimizer.state_dict()
            )

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()

        # train feature extractor and classifier by minimizing L_{err}
        for epoch in range(self.E0):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if len(data) <= 1:
                    continue
                self.optimizer.zero_grad()
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.classification_model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)

        for epoch in range(self.args.E1):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.discriminator.eval()
                self.generator.eval()
                data = data.to(self.device)
                target = target.to(self.device)
                y_onehot = torch.zeros(target.size(0), self.dataset.n_classes).to(self.device)
                y_onehot.scatter_(1, target.view(-1, 1), 0.6).to(self.device)
                randomn = torch.rand(target.size(0), self.generator.input_size).to(self.device)
                # train feature extractor and classifier by minimizing lambda_0*L_{advf}+lambda_1*L_{err}
                self.optimizer.zero_grad()
                feature = self.classification_model.base(data)
                y_pred = self.classification_model.classifier(feature)
                loss_err = criterion(y_pred, target)
                loss_advf = torch.mean(torch.pow(1 - self.discriminator(y_onehot, feature), 2))
                loss_cla = self.args.lambda_0 * loss_err + (1 - self.args.lambda_0) * loss_advf
                loss_cla.backward()
                self.optimizer.step()
                # train discriminator by minimizing L_{advd}
                self.disc_optimizer.zero_grad()
                self.classification_model.eval()
                self.discriminator.train()
                feature = self.classification_model.base(data)
                gen_feature = self.generator(y=y_onehot, x=randomn).detach()
                loss_advf = -torch.mean(
                    torch.pow(self.discriminator(y_onehot, gen_feature), 2)
                    + torch.pow(1 - self.discriminator(y_onehot, feature), 2)
                )
                loss_advf.backward()
                self.disc_optimizer.step()
                # train distribution generator by minimizing L_{advg}
                self.discriminator.eval()
                self.generator.train()
                self.gen_optimizer.zero_grad()
                gen_feature = self.generator(y=y_onehot, x=randomn).detach()
                loss_advg = torch.mean(torch.pow(1 - self.discriminator(y_onehot, gen_feature), 2))
                loss_advg.backward()
                self.gen_optimizer.step()

        self.classification_model.to(torch.device("cpu"))
        self.discriminator.to(torch.device("cpu"))
        self.generator.to(torch.device("cpu"))
        torch.cuda.empty_cache()
        self.logger.log(f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}")