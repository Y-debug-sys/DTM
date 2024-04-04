import os
import argparse
import numpy as np

from Train_utils.trainer import Trainer
from DDPM.preprocessing_model import PRE
from Utils.random_utils import setup_seed
from Utils.data_utils import build_dataloader
from DDPM.underlying_model import Transformer
from Utils.metric_utils import NMAE, NRMSE, MMD
from DDPM.diffusion_model import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser(description="Toy Experiment on Abilene Network", add_help=False)

    # args for random

    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training')
    parser.add_argument('--data_seed', type=int, default=123, 
                        help='seed for loading data')

    # args for dataset

    parser.add_argument('--data_root', type=str, default="./Data",
                        help="Root Dir of .csv File")
    parser.add_argument('--dataset', type=str, default="geant",
                        choices=["abilene", "geant"],
                        help="Dataset")
    parser.add_argument('--batch_size', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--feature_size', type=int, default=529, help='')
    parser.add_argument('--hidden_size', type=int, default=128, help='')
    parser.add_argument('--flow_known_rate', type=float, default=0.1,
                        help="Known Ratio (Link Loads during Testing)")
    parser.add_argument('--link_known_rate', type=float, default=0.,
                        help="Known Ratio (Link Loads during Testing)")
    parser.add_argument('--mode', type=str, default="TME",
                        choices=['TME', 'TMC', 'TMEC'],
                        help="Type of TM-related Tasks")
    
    # args for diffusion

    parser.add_argument('--self_condition', type=bool, default=False,
                        help='Use Self-Condition.')
    parser.add_argument('--time_steps', type=int, default=300,
                        help='Number of Diffusion Steps.')
    parser.add_argument('--sample_steps', type=int, default=300,
                        help='Number of Sampling Steps.')
    parser.add_argument('--loss_type', type=str, default='l1',
                        choices=['l1', 'l2'], help='Type of Loss Function.')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine'],
                        help='Type of Beta Schedule.')
    
    # args for training

    parser.add_argument('--base_lr', type=float, default=1e-5,
                        help='Learning Rate before Warmup.')
    parser.add_argument('--warmup_lr', type=float, default=8e-4,
                        help='Learning Rate after Warmup.')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='Minimum Learning Rate.')
    parser.add_argument('--warmup', type=int, default=500,
                        help='Number of Warmup Epochs.')
    parser.add_argument('--patience', type=int, default=2000,
                        help='Patience.')
    parser.add_argument('--threshold', type=float, default=1e-1,
                        help='Hyperparameter for Evaluating whether Better or not.')
    parser.add_argument('--factor', type=float, default=0.5,
                        help='Hyperparameter for Reducing Learning Rate.')
    parser.add_argument('--ema_cycle', type=int, default=10,
                        help='Number of Epochs between Two EMA Updating.')
    parser.add_argument('--ema_decay', type=float, default=0.995,
                        help='Decay Rate of EMA.')
    parser.add_argument('--train_epochs', type=int, default=10000,
                        help='Number of Training Epochs.')
    parser.add_argument('--save_cycle', type=int, default=1000, 
                        help='Number of Epochs between Two Model Saving.')
    parser.add_argument('--accumulate_cycle', type=int, default=2,
                        help='Number of Epochs between Two Gradient Descent.')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Train or Test.')

    args = parser.parse_args()
    return args


def main(args):
    setup_seed(args.seed)
    train_loader, test_loader, rm, rm_pinv = build_dataloader(data_root=args.data_root, dataset_name=args.dataset, flow_known_rate=args.flow_known_rate,
                                                              batch_size=args.batch_size, link_known_rate=args.link_known_rate, random_seed=args.data_seed,
                                                              mode=args.mode, train_size=3000, test_size=672, window=12)

    if args.flow_known_rate < 0.1:
        ratio = 0.5
    else:
        ratio = 0.
    
    tmc_model = PRE(train_loader.dataset.dim_2, output_size=args.feature_size, rm=rm, ratio=ratio)
    model = Transformer(n_feat=args.feature_size, n_layer_enc=2, n_layer_dec=2, n_embd=128,
                        n_heads=4, attn_pdrop=0., resid_pdrop=0., mlp_hidden_times=2, max_len=12)

    ddpm = GaussianDiffusion(model, timesteps=args.time_steps, sampling_timesteps=args.sample_steps, loss_type=args.loss_type,
                             objective='pred_x0', beta_schedule=args.beta_schedule, seq_length=12)

    trainer = Trainer(tmc_model, ddpm, train_loader, results_folder=f'./CPT/Checkpoints_{args.dataset}_{args.flow_known_rate}',
                      save_cycle=args.save_cycle, train_num_steps=args.train_epochs, patience=args.patience, min_lr=args.min_lr, 
                      threshold=args.threshold, warmup=args.warmup, factor=args.factor, warmup_lr=args.warmup_lr, 
                      gradient_accumulate_every=args.accumulate_cycle, train_lr=args.base_lr, ratio=args.flow_known_rate)

    if args.is_train:
        trainer.train()
    else:
        trainer.load(milestone=10)

    root_cur = './Output'
    os.makedirs(root_cur, exist_ok=True)
    
    if args.mode == 'TME':
        estimation, reals = trainer.estimate(test_loader, rm, rm_pinv)
        np.save(os.path.join(root_cur, f'reals_estimation_{args.dataset}.npy'), reals)
        np.save(os.path.join(root_cur, f'traffic_estimation_dtm_{args.dataset}_{args.flow_known_rate}.npy'), estimation)

        loss_nmae = np.abs(estimation - reals).sum() / np.abs(reals).sum()
        loss_nrmse = np.sqrt(np.square(estimation - reals).sum()) / np.sqrt(np.square(reals).sum())
        print(loss_nmae, loss_nrmse)

    elif args.mode == 'TMC':
        completion, masks, reals = trainer.complete(test_loader)
        np.save(os.path.join(root_cur, f'reals_completion_{args.dataset}.npy'), reals)
        np.save(os.path.join(root_cur, f'traffic_completion_dtm_{args.dataset}_{args.flow_known_rate}.npy'), completion)
        np.save(os.path.join(root_cur, f'traffic_mask_{args.dataset}_{args.flow_known_rate}.npy'), masks)

        masks = (masks > 0).reshape(masks.shape)
        nmae = NMAE(reals, completion, masks)
        nrmse = NRMSE(reals, completion, masks)
        mmd = MMD(reals, completion)
        print(nmae, nrmse, mmd)

    else:
        completion, masks, reals = trainer.combine(test_loader, rm, rm_pinv, coef=0.25)
        np.save(os.path.join(root_cur, f'reals_completion_{args.dataset}.npy'), reals)
        np.save(os.path.join(root_cur, f'traffic_completion_{args.dataset}_{args.flow_known_rate}_{args.link_known_rate}.npy'), completion)
        np.save(os.path.join(root_cur, f'traffic_mask_{args.dataset}_{args.flow_known_rate}.npy'), masks)
        
        masks = (masks > 0).reshape(masks.shape)
        nmae = NMAE(reals, completion, masks)
        nrmse = NRMSE(reals, completion, masks)
        mmd = MMD(reals, completion)
        print(nmae, nrmse, mmd)


if __name__ == '__main__':
    args = parse_args()
    main(args)