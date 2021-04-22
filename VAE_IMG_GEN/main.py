import numpy as np
from arg_def import parser
from solver import Solver


def main(args):
    np.random.seed(args.seed)
    net = Solver(args)
    if args.train:
        net.train()
    else:
        net._test_model()
        # net.viz_traverse(args.limit)
    print('**** Finished ****')


if __name__ == "__main__":

    naive_train_args = parser.parse_args(
        "--train True "
        "--vae_model betavaeh "
        "--beta 1 "
        "--image_size 128 "
        "--save_step 500 "
        "--gather_step 15 "
        "--max_iter 1500 "
        "--viz_on True "
        "--is_PID False ".split())

    naive_test_args = parser.parse_args(
        "--train False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--is_PID False ".split())

    pid_train_args = parser.parse_args(
        "--train True "
        "--viz_on False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--is_PID True "
        "--image_size 128 "
        "--save_step 500 "
        "--gather_step 15 "
        "--max_iter 1500 "
        "--z_dim 500 "
        "--viz_on True "
        "--display_step 10 ".split())

    pid_test_args = parser.parse_args(
        "--train False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--z_dim 50 "
        "--is_PID True ".split())

    agg_naive_train_args = parser.parse_args(
        "--train True "
        "--vae_model betavaeh "
        "--beta 1 "
        "--image_size 128 "
        "--save_step 500 "
        "--gather_step 15 "
        "--max_iter 1500 "
        "--aggressive True "
        "--viz_name agg_enc "
        "--viz_on True "
        "--is_PID False ".split())

    agg_pid_train_args = parser.parse_args(
        "--train True "
        "--viz_on False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--is_PID True "
        "--image_size 128 "
        "--save_step 500 "
        "--gather_step 15 "
        "--max_iter 1500 "
        "--z_dim 500 "
        "--viz_on True "
        "--aggressive True "
        "--viz_name agg_enc "
        "--display_step 10 ".split())

    agg_naive_test_args = parser.parse_args(
        "--train False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--z_dim 500 "
        "--aggressive True "
        "--viz_name agg_enc "
        "--is_PID True ".split())

    agg_pid_test_args = parser.parse_args(
        "--train False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--z_dim 500 "
        "--aggressive True "
        "--viz_name agg_enc "
        "--is_PID True ".split())

    main(agg_naive_test_args)
