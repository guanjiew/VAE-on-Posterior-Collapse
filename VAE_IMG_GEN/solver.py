import os
from tqdm import tqdm
import torch.cuda
import torch.optim as optim
from torch.distributions.normal import Normal
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torch.autograd import Variable
import visdom
import vae_model
from utils.DataGather import DataGather
from utils.helper import *
from preprocess.dataloader import load_data


def cuda(tensor, uses_cuda):
    return tensor.cuda() if uses_cuda else tensor


class Solver(object):
    def __init__(self, args):
        # Set up Env
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = args.max_iter
        self.global_iter = 0
        self.agg_iter = args.agg_iter
        # Hyper-parameters
        self.z_dim = args.z_dim
        self.batch_size = args.batch_size
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
        # Model architectures
        self.is_aggressive = args.aggressive
        self.is_PID = args.is_PID
        self.model_name = args.viz_name
        self.nc = 3
        self.decoder_dist = 'gaussian'
        if args.vae_model.endswith('skipvae'):
            self.model = 'skipvae'
        elif args.vae_model.endswith('betavaeh'):
            self.model = 'betavaeh'
        else:
            self.model = 'vae'
        if self.model == 'betavaeh' or self.model == 'skipvae':
            net = vae_model.__dict__[args.vae_model](
                z_dim=self.z_dim,
                nc=self.nc,
                is_classification=False,
                num_classes=None)
        else:
            raise NotImplementedError('only support vae_model H and skip vae')

        # load model and set up adam
        self.net = cuda(net, self.use_cuda)
        if self.model == 'skipvae':
            self.skip_optim = optim.Adam(self.net.vae.parameters(), lr=self.lr,
                                         betas=(self.beta1, self.beta2))
        else:
            self.enc_optim = optim.Adam(self.net.encoder.parameters(), lr=self.lr,
                                        betas=(self.beta1, self.beta2))
            self.dec_optim = optim.Adam(self.net.decoder.parameters(), lr=self.lr,
                                        betas=(self.beta1, self.beta2))

        # Visualization tasks
        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_beta = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        self.save_output = args.save_output
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port, use_incoming_socket=False)
        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step
        self.delta = args.delta
        # Define output / checkpoint directory
        self._set_path(args)

        # Set up dataset
        self.dset_dir = args.dset_dir
        self.dataset = args.dataset
        self.data_loader = load_data(args)
        self.gather = DataGather()
        self.gather2 = DataGather()

        self.args = args

        output = "****"
        if self.is_PID:
            output += "Control PID + "
        if self.model == "skipvae":
            output += " Skip VAE Model "
        else:
            output += " Beta VAE Model"
        output += " with beta = {:.3f} Solver Initialized ****".format(self.beta)
        print(output)

    def _set_path(self, args):
        self.output_dir_name = args.output_dir + '/'
        self.checkpt_dir_name = args.ckpt_dir + '/'
        self.result_dir_name = 'test-result/'
        if self.is_PID:
            self.output_dir_name += 'Control-'
            self.checkpt_dir_name += 'Control-'
            self.result_dir_name += 'Control-'
        if args.vae_model.endswith('skipvae'):
            self.output_dir_name += 'SkipVAE'
            self.checkpt_dir_name += 'SkipVAE'
            self.result_dir_name += 'SkipVAE'
        else:
            self.output_dir_name += 'BetaVAE'
            self.checkpt_dir_name += 'BetaVAE'
            self.result_dir_name += 'BetaVAE'

        self.output_dir = os.path.join(self.output_dir_name, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.test_path = os.path.join(self.result_dir_name, self.model_name)
        if not os.path.exists(self.test_path):
            os.makedirs(self.test_path, exist_ok=True)

        self.ckpt_dir = os.path.join(self.checkpt_dir_name, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            self.load_checkpoint(self.ckpt_name)

    def train(self):
        print('**** Start Training ****')
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        outfile = os.path.join(self.ckpt_dir, "train.log")
        agg_outfile = os.path.join(self.ckpt_dir, "agg_train.log")
        print('**** Start Logging ****')
        fw_log = open(outfile, "a")
        if self.args.aggressive:
            afw_log = open(agg_outfile, "a")
        if self.is_PID:
            PID = vae_model.__dict__["pid"]()

        Kp = 0.01
        Ki = -0.0001
        Kd = 0.0
        fw_log.write("Kp:{0:.5f} Ki: {1:.6f}\n".format(Kp, Ki))
        fw_log.flush()

        out = False
        while not out:
            for x in self.data_loader:
                if not self.args.is_classification:
                    x, _ = x
                self.global_iter += 1
                pbar.update(1)
                x = Variable(cuda(x, self.use_cuda))

                # Strengthen inference network: train encoder more aggressively than decoder
                sub_iter = 1
                while self.is_aggressive and sub_iter < self.agg_iter and self.model != 'skipvae':
                    print("*** Start Aggressive Training ***")
                    self.enc_optim.zero_grad()
                    self.dec_optim.zero_grad()
                    x_recon, mu, logvar = self.net(x)
                    recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                    total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
                    if self.is_PID:
                        self.beta, _ = PID.pid(self.KL_loss, total_kld.item(), Kp, Ki, Kd)
                        beta_vae_loss = recon_loss + self.beta * total_kld
                    else:
                        beta_vae_loss = recon_loss + 1.0 * total_kld
                    beta_vae_loss.backward()
                    self.enc_optim.step()
                    afw_log.write(
                        '[Global iter:{}][sub iter:{}] beta_vae_loss:{:.3f} recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} beta:{:.4f}\n'.format(
                            self.global_iter, sub_iter, beta_vae_loss.item(), recon_loss.item(), total_kld.item(),
                            mean_kld.item(), self.beta))
                    afw_log.flush()
                    print(f'recon loss during sub iter {sub_iter}, global iter {self.global_iter}: {recon_loss}')
                    sub_iter += 1
                    if sub_iter == self.agg_iter:
                        print("*** End Aggressive Training for {} iterations***".format(sub_iter))

                x_recon, mu, logvar = self.net.forward(x)
                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.is_PID:
                    self.beta, _ = PID.pid(self.KL_loss, total_kld.item(), Kp, Ki, Kd)
                    beta_vae_loss = recon_loss + self.beta * total_kld
                else:
                    beta_vae_loss = recon_loss + 1.0 * total_kld

                if self.model == 'skipvae':
                    self.skip_optim.zero_grad()
                    beta_vae_loss.backward()
                    self.skip_optim.step()
                else:
                    self.enc_optim.zero_grad()
                    self.dec_optim.zero_grad()
                    beta_vae_loss.backward()
                    if not self.is_aggressive:
                        self.enc_optim.step()
                    self.dec_optim.step()
                    # Criteria from the paper for setting aggressive to False
                    # only if all training data has been processed
                    if self.is_aggressive:
                        # TODO: replace criteria below with MUTUAL information --> using a validation data batch
                        # according to paper, mutual info criteria was usually achieved around approx 5 epochs
                        # running 6 here, at 6, the optimization doesnt lower loss by that much anymore
                        if self.global_iter > 5:
                            self.is_aggressive = False

                # Visualization and Logging
                if self.viz_on and self.global_iter % self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       mean_kld=mean_kld.data, beta=self.beta)
                    self.gather2.insert(iter=self.global_iter,
                                        mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                        recon_loss=recon_loss.data, total_kld=total_kld.data,
                                        mean_kld=mean_kld.data, beta=self.beta)
                    self.gather.insert(images=x.data)
                    self.gather.insert(images=torch.sigmoid(x_recon).data)
                    self.viz_reconstruction()
                    self.viz_lines()
                    self.gather.flush()

                if (self.viz_on or self.save_output) and self.global_iter % 10000 == 0:
                    if self.model == "skipvae":
                        self.viz_traverse_skip()
                    else:
                        self.viz_traverse()

                if self.global_iter % 20 == 0:
                    fw_log.write(
                        '[{}] beta_vae_loss:{:.3f} recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f} beta:{:.4f}\n'.format(
                            self.global_iter, beta_vae_loss.item(), recon_loss.item(), total_kld.item(),
                            mean_kld.item(), self.beta))
                    fw_log.flush()

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

    def test(self):
        print('****** Start testing model ****')

        predict_path = os.path.join(self.test_path, 'prediction')
        ground_path = os.path.join(self.test_path, 'ground_truth')
        image_path = [predict_path, ground_path]

        for path in image_path:
            if not os.path.exists(path):
                os.makedirs(path)

        self.net_mode(train=False)
        ids = 0
        batch = 0
        num_image = 5

        outfile = os.path.join(self.test_path, "test.log")
        fw_log = open(outfile, "w")

        epsilons = Variable(cuda(torch.arange(0, 1, 0.01), self.use_cuda))
        epsilons_size = epsilons.size(0)
        rep_epsilons = epsilons.view(epsilons_size, 1).repeat(1, self.z_dim)

        all_zs_lt_epsilon = torch.zeros(epsilons_size, self.z_dim).to(epsilons.device)

        for x in self.data_loader:
            if not self.args.is_classification:
                x, _ = x
            batch += 1
            x = Variable(cuda(x, self.use_cuda))
            x_recon, mu, logvar = self.net(x)
            recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)  # dim_wise_kld dim = z_dim

            fw_log.write('recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}\n'.format(
                recon_loss.item(), total_kld.item(), mean_kld.item()))
            fw_log.flush()

            zs_lt_epsilon = dim_wise_kld.repeat(epsilons_size, 1) < rep_epsilons
            all_zs_lt_epsilon += zs_lt_epsilon

            samples = F.sigmoid(x_recon).data
            batch_size = samples.size(0)

            for b in range(batch_size):
                ids += 1
                save_image(samples[b, :, :, :], fp=os.path.join(image_path[0], 'predict_{}.jpg'.format(ids)))
                save_image(x[b, :, :, :], fp=os.path.join(image_path[1], 'ground_{}.jpg'.format(ids)))

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
        prop_collapsed = prop_collapsed.view(epsilons_size, ).cpu().detach().numpy()

        plt.plot(epsilons, prop_collapsed)
        plt.xlabel('epsilon')
        plt.ylabel('collapse %')
        plt.title('Posterior collapse')
        fig.savefig(os.path.join(output_dir, 'graph_delta_epsilon_collapse_percentage.jpg'))
        
        # Mutual information
        kl1 = None
        z2 = None
        for x in self.data_loader:
            x = Variable(cuda(x, self.use_cuda))
            _, mean, logvar = self.net(x)
            logstd = logvar / 2
            dst = Normal(loc=mean, scale=torch.exp(logstd))
            z1 = dst.sample((N1,))
            batch_z2 = torch.cat(list(dst.sample((N2,))))
            if z2 is None:
                z2 = batch_z2
            else:
                z2 = torch.cat((z2, batch_z2))
            batch_kl1 = torch.mean(torch.sum(((z1 ** 2 - ((z1 - mean) / torch.exp(logstd)) ** 2) / 2) - logstd, dim=-1), dim=0)
            if kl1 is None:
                kl1 = batch_kl1
            else:
                kl1 = torch.cat((kl1, batch_kl1))
        kl1 = torch.mean(kl1)

        kl2 = None
        for x in self.data_loader:
            x = Variable(cuda(x, self.use_cuda))
            _, mean, logvar = self.net(x)
            logstd = logvar / 2
            mean = mean.unsqueeze(1)
            logstd = logstd.unsqueeze(1)
            batch_kl2 = torch.mean(torch.sum(((z2 ** 2 - ((z2 - mean) / torch.exp(logstd)) ** 2) / 2) - logstd, dim=-1), dim=-1)
            if kl2 is None:
                kl2 = batch_kl2
            else:
                kl2 = torch.cat((kl2, batch_kl2))
        kl2 = torch.mean(kl2)
        print("Mutual information:", (kl1 - kl2).item()) 

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net': self.net.state_dict()}
        win_states = {'recon': self.win_recon,
                      'beta': self.win_beta,
                      'kld': self.win_kld,
                      'mu': self.win_mu,
                      'var': self.win_var, }
        if self.model == "skipvae":
            skip_optim_states = {'skip_optim': self.skip_optim.state_dict()}
            states = {'iter': self.global_iter,
                      'win_states': win_states,
                      'skip_optim_states': skip_optim_states,
                      'model_states': model_states,
                      }
        else:
            enc_optim_states = {'enc_optim': self.enc_optim.state_dict(), }
            dec_optim_states = {'dec_optim': self.dec_optim.state_dict(), }
            states = {'iter': self.global_iter,
                      'win_states': win_states,
                      'model_states': model_states,
                      'enc_optim_states': enc_optim_states,
                      'dec_optim_states': dec_optim_states}
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
            if self.model == "skipvae":
                self.skip_optim.load_state_dict(checkpoint['skip_optim_states']['skip_optim'])
            else:
                self.enc_optim.load_state_dict(checkpoint['enc_optim_states']['enc_optim'])
                self.dec_optim.load_state_dict(checkpoint['dec_optim_states']['dec_optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))

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

    def viz_traverse_skip(self, limit=3, inter=2 / 3, loc=-1):
        self.net_mode(train=False)
        import random
        num_image = 7

        vae = self.net

        interpolation = torch.arange(-limit, limit + 0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets - 1)

        random_img = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_recon_x, _, _ = vae.forward(random_img)[:, :self.z_dim]

        ###------------fixed image------------------
        fixed_idx = 0
        fixed_img = self.data_loader.dataset.__getitem__(fixed_idx)
        fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
        fixed_recon_x, _, _ = vae.forward(fixed_img)[:, :self.z_dim]

        ## save image to folder
        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)

        ## visualize image
        X_image = {'fixed_x': fixed_recon_x, 'random_x': random_recon_x}

        for key in X_image.keys():
            img = X_image[key]
            samples = torch.sigmoid(img).data
            ## visualize
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

        ## visualize image
        Z_image = {'fixed_z': fixed_z, 'random_z': random_z}

        for key in Z_image.keys():
            z = Z_image[key]
            samples = torch.sigmoid(decoder(z)).data
            ## visualize
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

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise Exception('Only bool type is supported. True or False')
        if train:
            self.net.train()
        else:
            self.net.eval()
