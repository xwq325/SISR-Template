import os
import torch
import numpy as np
import random
import utility
import data
import model
import loss
from option import args
from trainer import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    checkpoint = utility.checkpoint(args)
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()
            if args.params_flops_idx_scale != -1 and args.params_flops_dataset != '':
                t.calc_params_flops()
            checkpoint.done()


if __name__ == '__main__':
    main()
