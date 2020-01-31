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
        log_every=None,
    ):
        self.ensemble = ensemble
        self.reward_model = reward_model
        self.buffer = buffer
        self.n_train_epochs = n_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.grad_clip_norm = grad_clip_norm
        self.log_every = log_every

        self.params = list(ensemble.parameters()) + list(reward_model.parameters())
        self.optim = torch.optim.Adam(self.params, lr=learning_rate, eps=epsilon)

    def train(self):
        message = "Training on {} data points"
        print(message.format(self.buffer.total_steps))

        for epoch in range(1, self.n_train_epochs + 1):
            e_losses = []
            r_losses = []
            for (states, actions, rewards, deltas) in self.buffer.get_train_batches(
                self.batch_size
            ):
                self.ensemble.train()
                self.reward_model.train()

                self.optim.zero_grad()
                e_loss = self.ensemble.loss(states, actions, deltas)
                r_loss = self.reward_model.loss(states, rewards)
                e_losses.append(e_loss.item())
                r_losses.append(r_loss.item())
                (e_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.params, self.grad_clip_norm, norm_type=2
                )
                self.optim.step()

            if self.log_every is not None and epoch % self.log_every == 0:
                message = "> Train epoch {} [ensemble {:.2f} | reward {:.2f}]"
                print(message.format(epoch, e_loss.item(), r_loss.item()))

        message = "Summed losses: [ensemble {:.2f} | reward {:.2f}]"
        print(message.format(sum(e_losses), sum(r_losses)))
