# Set self.beta to 1 and use PID to false for regular VAE
import os
from tqdm import tqdm
import torch.cuda
import torch.optim as optim
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import visdom
import vae_model
from DataGather import DataGather
from helper import *
from preprocess.dataloader import load_data


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


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
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.KL_loss = args.KL_loss
        self.pid_fixed = args.pid_fixed
        self.is_PID = args.is_PID
        self.model_name = args.viz_name
        self.nc = 3
        self.decoder_dist = 'gaussian'

        if args.vae_model.endswith('betavaeh'):
            net = vae_model.__dict__[args.vae_model](
                z_dim=self.z_dim,
                nc=self.nc,
                is_classification=False,
                num_classes=None)
        else:
            raise NotImplementedError('only support vae_model H')

        # load vae_model
        self.net = cuda(net, self.use_cuda)
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

        if self.is_PID:
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

        if self.is_PID:
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
        self.args = args

        if self.is_PID:
            print('**** Control PID + Beta VAE Model with beta = {:.3f} Solver Initialized ****'.format(self.beta))
        else:
            print('**** Beta VAE Model with beta = {:.3f} Solver Initialized ****'.format(self.beta))

    def train(self):
        print('**** Start Training ****')
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        outfile = os.path.join(self.ckpt_dir, "train.log")
        print('**** Start Logging ****')
        fw_log = open(outfile, "w")
        if self.is_PID:
            PID = vae_model.__dict__["pid"]()
        Kp = 0.01
        Ki = -0.0001
        Kd = 0.0
        fw_log.write("Kp:{0:.5f} Ki: {1:.6f}\n".format(Kp, Ki))
        fw_log.flush()
        while not out:
            for x in self.data_loader:
                if not self.args.is_classification:
                    x, _ = x
                self.global_iter += 1
                pbar.update(1)
                x = Variable(cuda(x, self.use_cuda))
                x_recon, mu, logvar = self.net(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                if self.is_PID:
                    self.beta, _ = PID.pid(self.KL_loss, total_kld.item(), Kp, Ki, Kd)
                    beta_vae_loss = recon_loss + self.beta * total_kld
                else:
                    beta_vae_loss = recon_loss + 1.0 * total_kld
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
                    fw_log.write(
                        '[{}] beta_vae_loss:{:.3f} recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} beta:{:.4f}\n'.format(
                            self.global_iter, beta_vae_loss.item(), recon_loss.item(), total_kld.item(),
                            mean_kld.item(), self.beta))
                    fw_log.flush()
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
        print('**** End Logging ****')
        print('**** End Training ****')

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
            save_image(tensor=images, fp=os.path.join(output_dir, 'recon.jpg'), pad_value=1)
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
                    title=' reconsturction loss', ))
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
                save_image(samples, fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, self.global_iter)), \
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

        outfile = os.path.join(test_path, "test.log")
        fw_log = open(outfile, "w")

        epsilons = Variable(cuda(torch.arange(0, 1, 0.01), self.use_cuda))
        epsilons_size = epsilons.size(0)
        rep_epsilons = epsilons.view(epsilons_size, 1).repeat(1, self.z_dim)

        all_zs_lt_epsilon = torch.zeros(epsilons_size,self.z_dim).to(epsilons.device)

        for x in self.data_loader:
            if not self.args.is_classification:
                x, _ = x
            batch += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, mu, logvar = self.net(x)
            recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar) # dim_wise_kld dim = z_dim

            fw_log.write('recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}\n'.format(
                            recon_loss.item(), total_kld.item(), mean_kld.item()))
            fw_log.flush()

            zs_lt_epsilon = dim_wise_kld.repeat(epsilons_size, 1) < rep_epsilons
            all_zs_lt_epsilon += zs_lt_epsilon

            samples = F.sigmoid(x_recon).data
            batch_size = samples.size(0)

            for b in range(batch_size):
                ids += 1
                save_image(samples[b,:,:,:], fp=os.path.join(image_path[0], 'predict_{}.jpg'.format(ids)))
                save_image(x[b,:,:,:], fp=os.path.join(image_path[1], 'ground_{}.jpg'.format(ids)))

        fw_log.close()

        prop_lt_epsilon = torch.div(all_zs_lt_epsilon, batch)
        collapsed = prop_lt_epsilon >= self.delta
        prop_collapsed = torch.div(torch.sum(collapsed, dim=1), self.z_dim)

        # Plot epsilon delta collapse graph for various epsilons
        output_dir = os.path.join(self.output_dir, 'delta_epsilon_collapse')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        fig = plt.figure(figsize=(10, 10), dpi=300)
        epsilons = epsilons.cpu().detach().numpy()
        prop_collapsed = prop_collapsed.view(epsilons_size,).cpu().detach().numpy()

        plt.plot(epsilons, prop_collapsed)
        plt.xlabel('epsilon')
        plt.ylabel('collapse %')
        plt.title('Posterior collapse')
        fig.savefig(os.path.join(output_dir, 'graph_delta_epsilon_collapse_percentage.jpg'))

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise Exception('Only bool type is supported. True or False')
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
