# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config_t_ref, load_config_a_ref
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.modeling import PtTransformer0
from libs.utils import valid_one_epoch_all, ANETdetection, fix_random_seed
from libs.modeling import Refinement_module
import pickle
import json

################################################################################
label_dic = {'CricketBowling': 5, 'CricketShot': 6,
                'VolleyballSpiking': 19, 'JavelinThrow': 12,
                'Shotput': 15, 'TennisSwing': 17, 'GolfSwing': 9,
                'ThrowDiscus': 18, 'Billiards': 2, 'CleanAndJerk': 3,
                'LongJump': 13, 'Diving': 7, 'CliffDiving': 4,
                'BasketballDunk': 1, 'HighJump': 11, 'BaseballPitch': 0,
                'HammerThrow': 10, 'SoccerPenalty': 16,
                'FrisbeeCatch': 8, 'PoleVault': 14}


def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg_ref = load_config_t_ref(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg_ref['val_split']) > 0, "Test set must be specified!"
    if ".pth.tar" in args.ckpt:
        assert os.path.isfile(args.ckpt), "CKPT file does not exist!"
        ckpt_file = args.ckpt
    else:
        assert os.path.isdir(args.ckpt), "CKPT file folder does not exist!"
        if args.epoch > 0:
            ckpt_file = os.path.join(
                args.ckpt, 'epoch_{:03d}.pth.tar'.format(args.epoch)
            )
        else:
            ckpt_file_list = sorted(glob.glob(os.path.join(args.ckpt, '*.pth.tar')))
            ckpt_file = ckpt_file_list[-1]
        assert os.path.exists(ckpt_file)

    if args.topk > 0:
        cfg_ref['model']['test_cfg']['max_seg_num'] = args.topk
    # pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg_ref['dataset_name'], False, cfg_ref['val_split'], **cfg_ref['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg_ref['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    ref_model = Refinement_module(**cfg_ref['model'])
    # not ideal for multi GPU training, ok for now
    ref_model = nn.DataParallel(ref_model, device_ids=cfg_ref['devices'])

    """4. load ckpt"""
    print("=> loading checkpoint '{}'".format(ckpt_file))
    # load ckpt, reset epoch / best rmse
    checkpoint = torch.load(
        ckpt_file,
        map_location = lambda storage, loc: storage.cuda(cfg_ref['devices'][0])
    )
    # load ema model instead
    # print("Loading from EMA model ...")
    ref_model.load_state_dict(checkpoint['state_dict_ema'])
    # ref_model = None
    del checkpoint

    # load TAD results
    TAD_path = "/cver/yfeng/project/20230318/RefineTAD/TAD results/AF_66.9.pkl"  # 66.87 67.75
    # TAD_path = "/cver/yfeng/project/20230318/RefineTAD/TAD results/thumos14_fusion_base.json"  # 51.28 53.27 53.69
    # TAD_path = "/cver/yfeng/project/20230318/RefineTAD/TAD results/detection_raw.json"  # 56.04 57.80 58.49
    if TAD_path.endswith(".pkl"):
        with open(TAD_path, "rb") as f:
            TAD_results = pickle.load(f)
    else:
        with open(TAD_path, "r", encoding="utf-8") as f:
            JSON_results = json.load(f)
        TAD_results = json2pkl(JSON_results)

    # set up evaluator
    det_eval, output_file = None, None
    if not args.saveonly:
        val_db_vars = val_dataset.get_attributes()
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
            # tiou_thresholds=[0.75,0.8,0.85,0.9,0.95]
        )

    else:
        output_file = os.path.join(os.path.split(ckpt_file)[0], 'eval_results.pkl')

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(TAD_path))
    start = time.time()
    mAP = valid_one_epoch_all(
        val_loader,
        TAD_results,
        ref_model,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg_ref['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        # refine=False,
        refine=True
    )
    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

def json2pkl(JSON_results):
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }
    for video_id, info in JSON_results['results'].items():
        for idx in range(len(info)):
            results['video-id'].append(video_id)
            results['t-start'].append(torch.Tensor([info[idx]['segment'][0]]))
            results['t-end'].append(torch.Tensor([info[idx]['segment'][1]]))
            results['label'].append(torch.Tensor([label_dic[info[idx]['label']]]))
            results['score'].append(torch.Tensor([info[idx]['score']]))
    results['t-start'] = torch.cat(results['t-start']).numpy()
    results['t-end'] = torch.cat(results['t-end']).numpy()
    results['label'] = torch.cat(results['label']).numpy()
    results['score'] = torch.cat(results['score']).numpy()
    return results

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('ckpt', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-epoch', type=int, default=-1,
                        help='checkpoint epoch')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)
