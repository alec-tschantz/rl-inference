# pylint: disable=not-callable
# pylint: disable=no-member

import torch


class Trainer(object):
    def __init__(
        self,
        ensemble,
        reward_model,
        buffer,
        n_train_epochs,
        batch_size,
        learning_rate,
        epsilon,
        grad_clip_norm,
        logger=None,
    ):
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.logger = logger

        self.params = list(ensemble.parameters()) + list(reward_model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def train(self):
        e_losses = []
        r_losses = []
        n_batches = []
        for epoch in range(1, self.n_train_epochs + 1):
            e_losses.append([])
            r_losses.append([])
            n_batches.append(0)

            for (states, actions, rewards, deltas) in self.buffer.get_train_batches(
                self.batch_size
            ):
                self.ensemble.train()
                self.reward_model.train()

                self.optim.zero_grad()
                e_loss = self.ensemble.loss(states, actions, deltas)
                r_loss = self.reward_model.loss(states, actions, rewards)
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.params, self.grad_clip_norm, norm_type=2
                )
                self.optim.step()

                e_losses[epoch - 1].append(e_loss.item())
                r_losses[epoch - 1].append(r_loss.item())
                n_batches[epoch - 1] += 1

            if self.logger is not None and epoch % 20 == 0:
                avg_e_loss = self._get_avg_loss(e_losses, n_batches, epoch)
                avg_r_loss = self._get_avg_loss(r_losses, n_batches, epoch)
                message = "> Train epoch {} [ensemble {:.2f} | reward {:.2f}]"
                self.logger.log(message.format(epoch, avg_e_loss, avg_r_loss))

        return (
            self._get_avg_loss(e_losses, n_batches, epoch),
            self._get_avg_loss(r_losses, n_batches, epoch),
        )

    def reset_models(self):
        self.ensemble.reset_parameters()
        self.reward_model.reset_parameters()
        self.params = list(self.ensemble.parameters()) + list(
            self.reward_model.parameters()
        )
        self.optim = torch.optim.Adam(
            self.params, lr=self.learning_rate, eps=self.epsilon
        )

    def _get_avg_loss(self, losses, n_batches, epoch):
        epoch_loss = [sum(loss) / n_batch for loss, n_batch in zip(losses, n_batches)]
        return sum(epoch_loss) / epoch
