# example from https://github.com/pytorch/examples/blob/master/vae/main.py
# commented and type annotated by Charl Botha <cpbotha@vxlabs.com>

import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision.models as models

from dataloader import EmojiDataset

# changed configuration to this instead of argparse for easier interaction
CUDA = torch.cuda.is_available()
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 20
EPOCHS = 20

# connections through the autoencoder bottleneck
Z_DIMS = 128

# torch.manual_seed(SEED)

# if CUDA:
#     torch.cuda.manual_seed(SEED)

# If you load your samples in the Dataset on CPU and would like to push it 
# during training to the GPU, you can speed up the host to device transfer by enabling pin_memory
pin_memory = CUDA

# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(
    EmojiDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=pin_memory)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        # encoder is pretrained resnet18 with parameter finetuning
        self.encoder = models.resnet18(pretrained=True)
        self.mu_fc = nn.Linear(1000, Z_DIMS)
        self.logvar_fc = nn.Linear(1000, Z_DIMS)

        # DECODER
        self.d1 = nn.Linear(Z_DIMS, 256*8*2*4*4)

        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd1 = nn.ReplicationPad2d(1)
        self.d2 = nn.Conv2d(256*8*2, 256*8, 3, 1)
        self.bn6 = nn.BatchNorm2d(256*8, 1.e-3)

        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd2 = nn.ReplicationPad2d(1)
        self.d3 = nn.Conv2d(256*8, 256*4, 3, 1)
        self.bn7 = nn.BatchNorm2d(256*4, 1.e-3)

        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd3 = nn.ReplicationPad2d(1)
        self.d4 = nn.Conv2d(256*4, 256*2, 3, 1)
        self.bn8 = nn.BatchNorm2d(256*2, 1.e-3)

        self.up4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.pd4 = nn.ReplicationPad2d(1)
        self.d5 = nn.Conv2d(256*2, 256, 3, 1)
        self.bn9 = nn.BatchNorm2d(256, 1.e-3)

        self.up5 = nn.UpsamplingNearest2d(scale_factor=4)
        self.pd5 = nn.ReplicationPad2d(1)
        self.d6 = nn.Conv2d(256, 3, 3, 1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        """Input vector x -> fully connected 1 -> ReLU -> (fully connected
        21, fully connected 22)

        Parameters
        ----------
        x : [128, 784] matrix; 128 digits of 28x28 pixels each

        Returns
        -------

        (mu, logvar) : Z_DIMS mean units one for each latent dimension, Z_DIMS
            variance units one for each latent dimension

        """

        resnet_output = self.encoder(x)
        mu = self.mu_fc(resnet_output)
        logvar = self.logvar_fc(resnet_output)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the Z_DIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, Z_DIMS] mean matrix
        logvar : [128, Z_DIMS] variance matrix

        Returns
        -------

        During training random sample from the learned Z_DIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            std = logvar.mul(0.5).exp_()
            if CUDA:
                eps = torch.cuda.FloatTensor(std.size()).normal_()
            else:
                eps = torch.FloatTensor(std.size()).normal_()

            eps = Variable(eps)
            return eps.mul(std).add_(mu)

        # During inference, we simply spit out the mean of the
        # learned distribution for the current input.  We could
        # use a random sample from the distribution, but mu of
        # course has the highest probability.
        return mu

    def decode(self, z):
        h1 = self.relu(self.d1(z))
        h1 = h1.view(-1, 256*8*2, 4, 4)
        h2 = self.leakyrelu(self.bn6(self.d2(self.pd1(self.up1(h1)))))
        h3 = self.leakyrelu(self.bn7(self.d3(self.pd2(self.up2(h2)))))
        h4 = self.leakyrelu(self.bn8(self.d4(self.pd3(self.up3(h3)))))
        h5 = self.leakyrelu(self.bn9(self.d5(self.pd4(self.up4(h4)))))

        return self.sigmoid(self.d6(self.pd5(self.up5(h5))))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

model = VAE()
if CUDA:
    model.cuda()

def loss_function(recon_x, x, mu, logvar):
    # how well do input x and output recon_x agree?
    BCE = F.binary_cross_entropy(recon_x, x)

    # KLD is Kullback–Leibler divergence -- how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # note the negative D_{KL} in appendix B of the paper
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= BATCH_SIZE * (256*256*3)

    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    return BCE + KLD

# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0.0

    # in the case of emoji dataset, len(train_loader.dataset) is ~3000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 128, 128]
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data)
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)

        # calculate scalar loss
        loss = loss_function(recon_batch, data, mu, logvar)

        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data / len(data)))

            save_image(data.data, '../results/progress/epoch_{}_batch_{}_data.png'.format(epoch, batch_idx), nrow=8, padding=2)
            save_image(recon_batch.data, '../results/progress/epoch_{}_batch_{}_recon.png'.format(epoch, batch_idx), nrow=8, padding=2)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


for epoch in range(1, EPOCHS + 1):
    train(epoch)

    if epoch == 1 or epoch % 5 == 0:
        torch.save(model.state_dict(), 'checkpoints/{}.pth'.format(epoch))

    # 64 sets of random Z_DIMS-float vectors, i.e. 64 emojis
    sample = Variable(torch.randn(9, Z_DIMS))
    if CUDA:
        sample = sample.cuda()
    print("decoding sample")
    sample = model.decode(sample).cpu()
    print('decoded')

    # save out as an 3x3 matrix of emojis
    # this will give you a visual idea of how well latent space can generate things
    # that look like digits
    save_image(sample.data, '../results/samples/epoch_{}.png'.format(epoch), nrow=3, padding=2)