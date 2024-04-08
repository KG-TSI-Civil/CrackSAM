import os
import sys
from tqdm import tqdm
import logging
import numpy as np
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import test_single_volume
from importlib import import_module
from segment_anything import sam_model_registry
from datasets.dataset_khanhha import Khanhha_dataset


def inference(args, multimask_output, db_config, model, test_save_path=None):
    db_test = db_config['Dataset'](base_dir=args.volume_path, list_dir=args.list_dir, split='test_vol')
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info(f'{len(testloader)} test iterations per epoch')
    model.eval()
    metric_list = 0.0
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        h, w = sampled_batch['image'].shape[2:]
        image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0] # tensor
        metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                      patch_size=[args.img_size, args.img_size], input_size=[args.input_size, args.input_size],
                                      test_save_path=test_save_path, case=case_name, z_spacing=db_config['z_spacing'])
        metric_list += np.array(metric_i)
        logging.info('idx %d case %s mean_pr %f mean_re %f mean_f1 %f  mean_iou %f' % (
            i_batch, case_name, metric_i[0], metric_i[1],metric_i[2], metric_i[3]))
    metric_list = metric_list / len(db_test)
    logging.info('Testing performance in best val model: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f' % (metric_list[0], metric_list[1],metric_list[2], metric_list[3]))
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for i in range(len(items)):
        key, value = items[i].strip().split(': ')
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str)
    parser.add_argument('--dataset', type=str, default='khanhha', help='Experiment name')
    parser.add_argument('--num_classes', type=int, default=1, help='For crack segmentation, the output class should be 1')
    parser.add_argument('--list_dir', type=str, default='./lists/lists_khanhha/', help='list_dir')
    parser.add_argument('--output_dir', type=str, default='/output')
    parser.add_argument('--img_size', type=int, default=448, help='Input image size of the network')
    parser.add_argument('--input_size', type=int, default=224, help='The input size for training SAM model')
    parser.add_argument('--seed', type=int,
                        default=3407, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_b_01ec64.pth',
                        help='Pretrained checkpoint')
    parser.add_argument('--delta_ckpt', type=str, default='checkpoints/epoch_159.pth', help='The checkpoint from adapter/LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--delta_type', type=str, default='choose from "adapter" or "lora" or "both"')
    parser.add_argument('--middle_dim', type=int, default=32, help='Middle dim of adapter')
    parser.add_argument('--scaling_factor', type=float, default=0.1, help='Scaling_factor of adapter')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')


    args = parser.parse_args()

    if args.config is not None:
        # overwtite default configurations with config file\
        config_dict = config_to_dict(args.config)
        for key in config_dict:
            setattr(args, key, config_dict[key])

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    dataset_config = {
        'khanhha': {
            'Dataset': Khanhha_dataset,
            'volume_path': args.volume_path,
            'list_dir': args.list_dir,
            'num_classes': args.num_classes,
            'z_spacing': 1
        }
    }
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                    num_classes=args.num_classes,
                                                                    checkpoint=args.ckpt, pixel_mean=[0, 0, 0], #load pre_trained backbone
                                                                     pixel_std=[1, 1, 1])
    if args.delta_type == 'adapter':
        pkg = import_module('delta.sam_adapter_image_encoder')
        net = pkg.Adapter_Sam(sam, args.middle_dim, args.scaling_factor).cuda()
    elif args.delta_type == 'lora':
        pkg = import_module('delta.sam_lora_image_encoder') 
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
    else:
        pkg = import_module('delta.sam_adapter_lora_image_encoder') 
        net = pkg.LoRA_Adapter_Sam(sam, args.middle_dim, args.rank).cuda()    

    assert args.delta_ckpt is not None
    net.load_delta_parameters(args.delta_ckpt) # load trained delta checkpoints

    if args.num_classes > 1:
        multimask_output = True
    else:
        multimask_output = False 

    # initialize log
    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder + '/' + 'log.txt', level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(test_save_path+ '/img/', exist_ok=True)
        os.makedirs(test_save_path+ '/pred/', exist_ok=True)
        os.makedirs(test_save_path+ '/gt/', exist_ok=True)
    else:
        test_save_path = None
    inference(args, multimask_output, dataset_config[dataset_name], net, test_save_path)

