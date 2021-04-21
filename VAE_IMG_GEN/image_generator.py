import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import os
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from math import exp
import visdom
from dataloader import load_data


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


############ Models ###########

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017).
        input_dim + 2*padding - k_size)/stride + 1
    """

    def __init__(self, z_dim=10, nc=3, is_classification=False, num_classes=None):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            ## in, out, kernel, stride, padding
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 64, 64
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  16,  16
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1),  # B, 256, 4, 4
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            View((-1, 256 * 4 * 4)),  # B, 256
            nn.Linear(256 * 4 * 4, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256 * 4 * 4),  # B, 256
            View((-1, 256, 4, 4)),  # B, 256,  4,  4
            nn.ReLU(True),
            ## stride *(n-1) + kernel - 2padding
            nn.ConvTranspose2d(256, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8*2,  8*2
            nn.BatchNorm2d(64, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16*2, 16*2
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            ##
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 64, 64
            nn.BatchNorm2d(32, 1.e-3),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 128, 128
        )

        self.is_classification = is_classification
        self.num_classes = num_classes

        # if is_classification:
        #     self.fc = nn.Linear(z_dim*2, num_classes)
        # else:
        #     self.fc = None
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        # if self.is_classification:
        #     classif = self.fc(distributions)
        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


########## Control VAE ###############

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = torch.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class PIDControl():
    """docstring for ClassName"""

    def __init__(self):
        """define them out of loop"""
        # self.exp_KL = exp_KL
        self.I_k1 = 0.0
        self.W_k1 = 0.0
        self.e_k1 = 0.0

    def _Kp_fun(self, Err, scale=1):
        return 1.0 / (1.0 + float(scale) * exp(Err))

    def pid(self, exp_KL, kl_loss, Kp=0.001, Ki=-0.001, Kd=0.01):
        """
        position PID algorithm
        Input: KL_loss
        return: weight for KL loss, beta
        """
        error_k = exp_KL - kl_loss
        ## comput U as the control factor
        Pk = Kp * self._Kp_fun(error_k)
        Ik = self.I_k1 + Ki * error_k
        # Dk = (error_k - self.e_k1) * Kd

        ## window up for integrator
        if self.W_k1 < 0 and self.W_k1 >= 1:
            Ik = self.I_k1

        Wk = Pk + Ik
        self.W_k1 = Wk
        self.I_k1 = Ik
        self.e_k1 = error_k

        ## min and max value
        if Wk > 1:
            Wk = 1.0
        if Wk < 0:
            Wk = 0.0

        return Wk, error_k


class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[], beta=[])

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


# Set self.beta to 1 and use PID to false for regular VAE
class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_max_org = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.KL_loss = args.KL_loss
        self.pid_fixed = args.pid_fixed
        self.is_PID = args.is_PID
        self.model_name = args.viz_name

        if args.dataset.lower() == 'celeba' or args.dataset.lower() == 'medmnist':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        else:
            raise NotImplementedError('only support model H')
        ## load model
        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                betas=(self.beta1, self.beta2))

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port, use_incoming_socket=False)

        if args.is_PID:
            checkpt_dir_name = args.ckpt_dir + '-ControlVAE'
        else:
            checkpt_dir_name = args.ckpt_dir

        self.ckpt_dir = os.path.join(checkpt_dir_name, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

        self.save_output = args.save_output

        if args.is_PID:
            output_dir_name = args.output_dir + '-ControlVAE'
        else:
            output_dir_name = args.output_dir

        self.output_dir = os.path.join(output_dir_name, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = load_data(args)

        self.gather = DataGather()
        self.gather2 = DataGather()

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        ## write log to log file
        outfile = os.path.join(self.ckpt_dir, "train.log")
        fw_log = open(outfile, "w")

        ## init PID control
        PID = PIDControl()
        Kp = 0.01
        Ki = -0.0001
        Kd = 0.0
        fw_log.write("Kp:{0:.5f} Ki: {1:.6f}\n".format(Kp, Ki))
        fw_log.flush()

        while not out:
            for x in self.data_loader:
                if not args.is_classification:
                    x, _ = x
                # print('shape>>', x.size())
                self.global_iter += 1
                pbar.update(1)

                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.is_PID:
                    # print(self.beta)
                    self.beta, _ = PID.pid(self.KL_loss, total_kld.item(), Kp, Ki, Kd)
                    beta_vae_loss = recon_loss + self.beta * total_kld
                else:
                    beta_vae_loss = recon_loss + 1.0 * total_kld
                    ### tricks for C
                    # C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    # beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                # print(f'Reconstruction loss is:{beta_vae_loss}')
                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       mean_kld=mean_kld.data, beta=self.beta)
                    self.gather2.insert(iter=self.global_iter,
                                        mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                        recon_loss=recon_loss.data, total_kld=total_kld.data,
                                        mean_kld=mean_kld.data, beta=self.beta)

                if self.global_iter % 20 == 0:
                    ## write log to file
                    fw_log.write(
                        '[{}] beta_vae_loss:{:.3f} recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} beta:{:.4f}\n'.format(
                            self.global_iter, beta_vae_loss.item(), recon_loss.item(), total_kld.item(),
                            mean_kld.item(), self.beta))
                    fw_log.flush()

                # if self.global_iter%self.display_step == 0:
                # pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} beta:{:.4f}'.format(
                # self.global_iter, recon_loss.item(), total_kld.item(), mean_kld.item(), self.beta))

                if self.viz_on and self.global_iter % self.save_step == 0:
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=torch.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    self.viz_lines()
                    self.gather.flush()

                if (self.viz_on or self.save_output) and self.global_iter % 50000 == 0:
                    self.viz_traverse()

                if self.global_iter % self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter % 20000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training for VAE Finished]")
        pbar.close()
        fw_log.close()

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:100]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:100]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name + '_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=10)
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            save_image(tensor=images, filename=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()
        betas = torch.Tensor(self.gather.data['beta'])

        # mus = torch.stack(self.gather.data['mu']).cpu()
        # vars = torch.stack(self.gather.data['var']).cpu()

        # dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        # klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        klds = torch.cat([mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        recon_losses_2 = torch.stack(self.gather2.data['recon_loss']).cpu()
        betas_2 = torch.Tensor(self.gather2.data['beta'])

        # mus_2 = torch.stack(self.gather2.data['mu']).cpu()
        # vars_2 = torch.stack(self.gather2.data['var']).cpu()

        # dim_wise_klds_2 = torch.stack(self.gather2.data['dim_wise_kld'])
        mean_klds_2 = torch.stack(self.gather2.data['mean_kld'])
        total_klds_2 = torch.stack(self.gather2.data['total_kld'])
        klds_2 = torch.cat([mean_klds_2, total_klds_2], 1).cpu()
        iters_2 = torch.Tensor(self.gather2.data['iter'])

        legend = []
        # for z_j in range(self.z_dim):
        #     legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                X=iters,
                Y=recon_losses,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='reconsturction loss', ))
        else:
            self.win_recon = self.viz.line(
                X=iters,
                Y=recon_losses,
                env=self.viz_name + '_lines',
                win=self.win_recon,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='reconsturction loss', ))

        if self.win_beta is None:
            self.win_beta = self.viz.line(
                X=iters,
                Y=betas,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='beta', ))
        else:
            self.win_beta = self.viz.line(
                X=iters,
                Y=betas,
                env=self.viz_name + '_lines',
                win=self.win_beta,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    xlabel='iteration',
                    title='beta', ))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                X=iters,
                Y=klds,
                env=self.viz_name + '_lines',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend,
                    xlabel='iteration',
                    title='kl divergence', ))
        else:
            self.win_kld = self.viz.line(
                X=iters,
                Y=klds,
                env=self.viz_name + '_lines',
                win=self.win_kld,
                update='append',
                opts=dict(
                    width=400,
                    height=400,
                    legend=legend,
                    xlabel='iteration',
                    title='kl divergence', ))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, recon_losses_2)
            plt.xlabel('iteration')
            plt.title('reconsturction loss')
            fig.savefig(os.path.join(output_dir, 'graph_recon_loss.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, betas_2)
            plt.xlabel('iteration')
            plt.title('beta')
            fig.savefig(os.path.join(output_dir, 'graph_beta.jpg'))

            fig = plt.figure(figsize=(10, 10), dpi=300)
            plt.plot(iters_2, klds_2)
            plt.legend(legend)
            plt.xlabel('iteration')
            plt.title('kl divergence')
            fig.savefig(os.path.join(output_dir, 'graph_kld.jpg'))

        self.net_mode(train=True)

    def viz_traverse(self, limit=3, inter=2 / 3, loc=-1):
        self.net_mode(train=False)
        import random
        num_image = 7

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit + 0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets - 1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_mu_z = encoder(random_img)[:, :self.z_dim]

        ###------------fixed image------------------
        fixed_idx = 0
        fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_mu_z = encoder(fixed_img)[:, :self.z_dim]
        # Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        eps = Variable(cuda(torch.FloatTensor(num_image, self.z_dim).uniform_(-1, 1), self.use_cuda), volatile=True)
        fixed_z = fixed_mu_z + eps

        ## ------------rand traverse------------------
        ## random hidden state from uniform
        random_z = Variable(cuda(torch.rand(num_image, self.z_dim), self.use_cuda), volatile=True)
        # random_z = Variable(cuda(torch.FloatTensor(1, self.z_dim).uniform_(-1, 1), self.use_cuda),volatile=True)

        ## save image to folder
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)

        ## visulize image
        Z_image = {'fixed_z': fixed_z, 'random_z': random_z}

        for key in Z_image.keys():
            z = Z_image[key]
            samples = torch.sigmoid(decoder(z)).data
            ## visulize
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)
            if self.viz_on:
                self.viz.images(samples, env=self.viz_name + '_traverse',
                                opts=dict(title=title), nrow=num_image)
            ## save image to folder
            if self.save_output:
                save_image(samples, filename=os.path.join(output_dir, '{}_{}.jpg'.format(key, self.global_iter)), \
                           nrow=num_image, pad_value=1)
        ###-------interplote linear space----------

        self.net_mode(train=True)

    def _test_model(self):
        print('******--testing model now--****')
        if self.is_PID:
            result_dir_name = 'results-ControlVAE'
        else:
            result_dir_name = 'results'
        test_path = os.path.join(result_dir_name, self.model_name)
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        predict_path = os.path.join(test_path, 'testing/predict-25')
        ground_path = os.path.join(test_path, 'ground/ground_truth-25')
        image_path = [predict_path, ground_path]

        for path in image_path:
            if not os.path.exists(path):
                os.makedirs(path)
        ## evaluate the result
        self.net_mode(train=False)
        ids = 0
        batch = 0
        num_image = 5
        for x in self.data_loader:
            if not args.is_classification:
                x, _ = x
            batch += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, _, _ = self.net(x)
            samples = torch.sigmoid(x_recon).data
            batch_size = samples.size(0)
            # save_image(samples, filename=os.path.join(image_path[0], 'recontruct_{}.eps'.format(batch)),nrow=num_image,pad_value=1)
            # save_image(x, filename=os.path.join(image_path[1], 'recontruct_{}.eps'.format(batch)),nrow=num_image,pad_value=1)
            # if batch >= 10:
            #     break
            for b in range(batch_size):
                ids += 1
                save_image(samples[b, :, :, :], fp=os.path.join(image_path[0], 'predict_{}.jpg'.format(ids)))
                save_image(x[b, :, :, :], fp=os.path.join(image_path[1], 'ground_{}.jpg'.format(ids)))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise ('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict(), }
        optim_states = {'optim': self.optim.state_dict(), }
        win_states = {'recon': self.win_recon,
                      'beta': self.win_beta,
                      'kld': self.win_kld,
                      'mu': self.win_mu,
                      'var': self.win_var, }
        states = {'iter': self.global_iter,
                  'win_states': win_states,
                  'model_states': model_states,
                  'optim_states': optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))


import argparse
import numpy as np
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# torch.cuda.set_device(0)

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    seed = args.seed
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    net = Solver(args)

    if args.train:
        net.train()
        # net._test_model()
    else:
        net._test_model()
        # net.viz_traverse(args.limit)
    print('****well done****')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Beta-VAE Model')
    parser.add_argument('--dataset', default='medmnist', type=str, help='dataset name', choices=['celeba', 'medmnist'])
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')

    parser.add_argument('--max_iter', default=1e6, type=float, help='maximum training iteration')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--limit', default=3, type=float, help='traverse limits')
    parser.add_argument('--KL_loss', default=25, type=float, help='KL KL_divergence')
    parser.add_argument('--pid_fixed', default=False, type=str2bool, help='if fixed PID or dynamic')
    parser.add_argument('--is_PID', default=True, type=str2bool, help='if use pid or not')
    parser.add_argument('--z_dim', default=100, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=4, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=5e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='data', type=str, help='dataset directory')

    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=10000, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=10000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=500, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--is_classification', default=False, type=bool, help='whether we are doing classification')

    args = parser.parse_args(
        "--train False "
        "--viz_on False "
        "--dataset celeba "
        "--seed 1 "
        "--lr 1e-4 "
        "--beta1 0.9 "
        "--beta2 0.999 "
        "--objective H "
        "--model H "
        "--batch_size 16 "
        "--z_dim 500 "
        "--max_iter 2000 "
        "--beta 1 "
        "--is_PID False "
        "--KL_loss 200 "
        "--image_size 128"
        "--save_step 10"
        "--gather_step 10"
        " --display_step 10".split())

    main(args)
