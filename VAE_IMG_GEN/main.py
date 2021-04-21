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
        "--save_step 10 "
        "--gather_step 10 "
        " --display_step 10 ".split())

    pid_test_args = parser.parse_args(
        "--train False "
        "--vae_model betavaeh "
        "--beta 1 "
        "--is_PID True ".split())

    main(naive_train_args)
