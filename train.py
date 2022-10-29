import argparse
import os
import data
import models as m
import torch
import torch.optim as optim
import yaml
from logger import Logger, ModelLogger
from trainers import *
import sys
import numpy as np
#from trainers import *

def build_model(model_type, args, device):
    model = getattr(m, model_type)(**args)
    return model.to(device)


def build_optimizer(opt_type, model, args):
    args['params'] = model.parameters()
    opt_class = getattr(optim, opt_type)
    return opt_class(**args)


def build_loggers(log_folder):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    model_logger = ModelLogger(log_folder, torch.save)
    scalar_logger = Logger(log_folder)
    return scalar_logger, model_logger


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='Yaml config with training parameters', required=True)
    parser.add_argument('--log_folder', '-l', help='Log folder', default="logs_reference_P")
    parser.add_argument('--data_root', '-r', help='Data root folder', required=False)
    parser.add_argument('--log_interval', '-i', type=int, help='Log interval', required=True)
    parser.add_argument('--resume', '-rs', type=str, help='resume', required=False)
    parser.add_argument('--projectname', type=str, help='name of the project_', required=True)
    parser.add_argument('--logpath', type=str, help='path where your training happens',default = 'logs')
    
    parser.add_argument('--perception_loss_weight', help='reconstruction_weight', default=6.0)
    parser.add_argument('--reconstruction_weight', help='reconstruction_weight', default=4.)
    parser.add_argument('--adversarial_weight', help='adversarial_weight', default=0.5)
    parser.add_argument('--append_smoothers', type=str, help='movement smoothing',default = 'True', choices=['True', 'False'])
    parser.add_argument('--filters_layers', type=str, help='test',default = '326412812812864', choices=['326412812812864', '323232323232','326464646464' ])
    parser.add_argument('--patch_size', type=str, help='test',default = '16', choices=[ '16','32','64','128'])
    parser.add_argument('--use_normalization', type=str, help='test',default = 'False', choices=['True', 'False'])
    parser.add_argument('--use_image_loss', type=str, help='test',default = 'True', choices=['True', 'False'])
    parser.add_argument('--tanh', type=str, help='test',default = 'True', choices=['True', 'False'])
    parser.add_argument('--use_bias', type=str, help='test',default = 'True', choices=['True', 'False'])
    args = parser.parse_args()

    perception_loss_weight = args.perception_loss_weight
    adversarial_weight = args.adversarial_weight
    reconstruction_weight = args.reconstruction_weight 
    perception_loss_weight2 = perception_loss_weight.replace( "'", "",2)
    adversarial_weight2 = adversarial_weight.replace( "'", "",2)
    reconstruction_weight2 = reconstruction_weight.replace( "'", "",2)
    
    #yaml config changer
    import yaml

    fname = args.config

    import pathlib
    from ruamel.yaml import YAML
    import ruamel
    #import ruamel.yaml
    def set_state(state):
        #yaml =  ruamel.yaml.YAML()
        yaml = YAML()
        mf = pathlib.Path(fname)
        yaml.default_flow_style = True
        doc = yaml.load(mf)#, Loader=ruamel.yaml.RoundTripLoader)#, ruamel.yaml.preserve_quotes=False)
        if args.append_smoothers == 'False':
            doc['generator']['args']['append_smoothers'] = False
        elif args.append_smoothers == 'True':
            doc['generator']['args']['append_smoothers'] = True
        if args.filters_layers == '326412812812864':    
            doc['generator']['args']['filters']= [32, 64, 128, 128, 128, 64]
        elif args.filters_layers == '323232323232':    
            doc['generator']['args']['filters']= [32, 32, 32, 32, 32, 32]
        elif args.filters_layers == '323232323232':    
            doc['generator']['args']['filters']= [32, 32, 32, 32, 32, 32]
        elif args.filters_layers == '326464646464' :  
            doc['generator']['args']['filters']= [32, 64, 64, 64, 64, 64]
        if args.patch_size == '128':
            doc['training_dataset']['patch_size']= 128
        elif args.patch_size == '16':
            doc['training_dataset']['patch_size']= 16
        elif args.patch_size == '32':
            doc['training_dataset']['patch_size']= 32
        elif args.patch_size == '64':
            doc['training_dataset']['patch_size']= 64
        if args.use_normalization == 'False':
            doc['perception_loss']['perception_model']['args']['use_normalization'] = False
        elif args.use_normalization == 'True':
            doc['perception_loss']['perception_model']['args']['use_normalization'] = True
        if args.use_image_loss == 'False':
            doc['trainer']['use_image_loss'] = False
        elif args.use_image_loss == 'True':
            doc['trainer']['use_image_loss'] = True 
        if args.tanh == 'False':
            doc['generator']['args']['tanh'] = False
        elif args.tanh == 'True':
            doc['generator']['args']['tanh'] = True
        if args.use_bias == 'False':
            doc['generator']['args']['use_bias'] = False
        elif args.use_bias == 'True':
            doc['generator']['args']['use_bias'] = True
        doc['trainer']['adversarial_weight'] = [adversarial_weight]
        doc['trainer']['reconstruction_weight'] = [reconstruction_weight]
        #doc['trainer']['adversarial_weight'] = 0.5
        #doc['trainer']['reconstruction_weight'] = 40.
        yaml.dump(doc, mf)
    set_state(2)
    ####################################################
    
    """import sys
    from ruamel.yaml import YAML
    import pathlib

    fname = args.config
    mf = pathlib.Path(fname)
    yaml1 = YAML()
    data = yaml1.load(yaml_str)

    yaml1.dump(data, mf)"""
    
    if args.logpath:
        data_path = args.logpath
    else:
        data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')
    #data_path = os.path.expanduser('~\Documents\\visionsofchaos\\fewshot\\data')
    data_root = data_path + "\\" + args.projectname+"_train"
    args_log_folder = data_root + "/" + "logs_reference_P"

    with open(args.config, 'r') as f:
        job_description = yaml.load(f, Loader=yaml.FullLoader)
    
    
    config = job_description['job']
    
    scalar_logger, model_logger = build_loggers(args_log_folder)

    model_logger.copy_file(args.config)
    device = config.get('device') or 'cpu'

    # Check 'training_dataset' parameters
    training_dataset_parameters = set(config['training_dataset'].keys()) - \
                    {"type", "dir_pre", "dir_post", "dir_mask", "patch_size", "dir_x1", "dir_x2", "dir_x3", "dir_x4", "dir_x5", "dir_x6", "dir_x7", "dir_x8", "dir_x9", }
    if len(training_dataset_parameters) > 0:
        raise RuntimeError("Got unexpected parameter in training_dataset: " + str(training_dataset_parameters))

    d = dict(config['training_dataset'])
    d['dir_pre'] = data_root + "/" + d['dir_pre']
    d['dir_post'] = data_root + "/" + d['dir_post']
    d['device'] = config['device']
    if 'dir_mask' in d:
        d['dir_mask'] = data_root + "/" + d['dir_mask']

    # complete dir_x paths and set a correct number of channels
    channels = 3
    for dir_x_index in range(1, 10):
        dir_x_name = f"dir_x{dir_x_index}"
        d[dir_x_name] = data_root + "/" + d[dir_x_name] if dir_x_name in d else None
        channels = channels + 3 if d[dir_x_name] is not None else channels
    config['generator']['args']['input_channels'] = channels

    print(d)
    resumedata = str(args.resume)
    generator = build_model(config['generator']['type'], config['generator']['args'], device)
    if args.resume:
        generator = (torch.load(data_root + "/"+"/logs_reference_P"+"/"+resumedata+".pth", map_location=lambda storage, loc: storage)).to(device)
    opt_generator = build_optimizer(config['opt_generator']['type'], generator, config['opt_generator']['args'])

    discriminator, opt_discriminator = None, None
    if 'discriminator' in config:
        discriminator = build_model(config['discriminator']['type'], config['discriminator']['args'], device)
        #discriminator = (torch.load(args.data_root + "/disc_00300_style2.pth", map_location=lambda storage, loc: storage)).to(device)
        opt_discriminator = build_optimizer(config['opt_discriminator']['type'], discriminator, config['opt_discriminator']['args'])

    if 'type' not in d:
        raise RuntimeError("Type of training_dataset must be specified!")

    dataset_type = getattr(data, d.pop('type'))
    training_dataset = dataset_type(**d)

    train_loader = torch.utils.data.DataLoader(training_dataset, config['trainer']['batch_size'], shuffle=False,
                                               num_workers=config['num_workers'], drop_last=True)#, worker_init_fn=worker_init_fn)

    reconstruction_criterion = getattr(torch.nn, config['trainer']['reconstruction_criterion'])()
    adversarial_criterion = getattr(torch.nn, config['trainer']['adversarial_criterion'])()

    perception_loss_model = None
    perception_loss_weight = 1
    if 'perception_loss' in config:
        if 'perception_model' in config['perception_loss']:
            perception_loss_model = build_model(config['perception_loss']['perception_model']['type'],
                                                config['perception_loss']['perception_model']['args'],
                                                device)
        else:
            perception_loss_model = discriminator

        perception_loss_weight = int(float(perception_loss_weight2))
    adversarial_weight3 = int(float(adversarial_weight2))
    reconstruction_weight3 = int(float(reconstruction_weight2))
    trainer = Trainer(
        train_loader=train_loader,
        data_for_dataloader=d,  # data for later dataloader creation, if needed
        opt_generator=opt_generator, opt_discriminator=opt_discriminator,
        adversarial_criterion=adversarial_criterion, reconstruction_criterion=reconstruction_criterion,
        reconstruction_weight=adversarial_weight3,#config['trainer']['reconstruction_weight'],#args.reconstruction_weight,#config['trainer']['reconstruction_weight'],
        adversarial_weight=reconstruction_weight3,#config['trainer']['adversarial_weight'],#args.adversarial_weight,#config['trainer']['adversarial_weight'],
        log_interval=args.log_interval,
        model_logger=model_logger, scalar_logger=scalar_logger,
        perception_loss_model=perception_loss_model,
        perception_loss_weight=perception_loss_weight,
        use_image_loss=config['trainer']['use_image_loss'],
        device=device
    )

    args_config = args.config.replace('\\', '/')
    args_config = args_config[args_config.rfind('/') + 1:]
    trainer.train(generator, discriminator, int(config['trainer']['epochs']), data_root, args_config, 0)
    print("Training finished", flush=True)
    sys.exit(0)
