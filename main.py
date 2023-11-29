import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.HINT import HINT
import wandb

def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """
    config = load_config(mode)
    with wandb.init(project='Rstormer', config=load_config(mode)):

        # cuda visble devices
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


        # init device
        if torch.cuda.is_available():
            print('Cuda is available')
            config.DEVICE = torch.device("cuda")
            torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
        else:
            print('Cuda is unavailable, use cpu')
            config.DEVICE = torch.device("cpu")



        # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
        cv2.setNumThreads(0)


        # initialize random seed
        torch.manual_seed(config.SEED)
        torch.cuda.manual_seed_all(config.SEED)
        np.random.seed(config.SEED)
        random.seed(config.SEED)



        # build the model and initialize
        model = HINT(config)
        model.load()


        # model training
        if config.MODE == 1:
            config.print()
            print('\nstart training...\n')
            model.train()

        # model test
        elif config.MODE == 2:
            print('\nstart testing...\n')
            model.test()




def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '--checkpoints', type=str, default='',
                        help='model checkpoints path (default: ./checkpoints)')

    parser.add_argument('--model', type=int, default='2', choices=[2])

    # test mode
    if mode == 2:
        parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
        parser.add_argument('--output', type=str, help='path to the output directory')

    args = parser.parse_args()
    config_path = os.path.join(args.path, 'config.yml')
    # config_path = os.path.join(args.path, 'config_test.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(args.path):
        os.makedirs(args.path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)
    print(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

    # test mode
    elif mode == 2:
        config.MODE = 2
        config.MODEL = args.model if args.model is not None else 3

        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.output is not None:
            config.RESULTS = args.output


    return config


if __name__ == "__main__":
    main()
