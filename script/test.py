import os
import csv
import torch

from script.validate import validate
from options.test_options import TestOptions

import warnings
warnings.filterwarnings('ignore')

opt = TestOptions().parse(print_options=False)
rows = [["Model: {}".format(opt.name)],
        ["Path: {}".format(opt.model_path)],
        ['TestSet', 'ACC', 'AP', 'r_ACC', 'f_ACC', 'AUC']]


# Model
assert opt.model_path is not None

if opt.detection_model == 'LGrad':
    from models.modules.LGrad.resnet import resnet50
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)
elif opt.detection_model == 'LNP':
    from models.modules.LNP.resnet import resnet50
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)
elif opt.detection_model == 'GramNet':
    from models.modules.GramNet.resnet_gram import resnet18
    model = resnet18(num_classes=1)
elif opt.detection_model == 'FreDect':
    import torchvision
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(2048, 1)
elif opt.detection_model == 'CNNSpot':
    from models.modules.CNNSpot.resnet50 import resnet50
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Linear(2048, 1)
elif opt.detection_model == 'UnivFD':
    from models.modules.UnivFD import get_model
    model = get_model(opt.arch)
elif opt.detection_model == 'NPR':
    from models.modules.NPR.resnet import resnet50
    model = resnet50(pretrained=False, num_classes=1)
elif opt.detection_model == 'SAFE':
    from models.modules.SAFE.resnet import resnet50
    model = resnet50(num_classes=2)
elif opt.detection_model == 'SimE':
    from models.modules.SimE import load_model
    model, _ = load_model(opt.arch, device='cpu')
elif opt.detection_model == 'FatFormer':
    from models.modules.FatFormer import build_model
    model = build_model(opt)
else:
    raise ValueError(f'{opt.detection_model} not supported for now')

state_dict = torch.load(opt.model_path, map_location='cpu')
try:
    if opt.detection_model in ['GramNet', 'FreDect']:
        try:
            model.load_state_dict(state_dict['netC'], strict=True)
        except:
            model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()}, strict=True)
    elif opt.detection_model == 'UnivFD':
        model.fc.load_state_dict(state_dict)
    elif opt.detection_model == 'NPR':
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()}, strict=True)
    elif opt.detection_model == 'SimE':
        model.load_weights(opt.model_path)
    else:
        model.load_state_dict(state_dict['model'], strict=True)
except:
    # for re-implemented models
    model.load_state_dict(state_dict['model'], strict=True)

print('Loading checkpoints from {} successfully'.format(opt.model_path))

model.to(opt.device)
model.eval()

# Testset
if opt.test_dataset == 'ProGAN':
    dataroot = './datasets/ProGAN/test'
    vals = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'stylegan2', 'deepfake']
    multiclasses = [1, 1, 0, 1, 0, 0, 1, 0]
elif opt.test_dataset == 'LSUN-bedroom':
    dataroot = './datasets/LSUN-bedroom'
    vals = ['DDPM', 'ADM', 'IDDPM', 'LDM', 'PNDM']
    multiclasses = [0]*len(vals)
elif opt.test_dataset == 'GenImage':
    dataroot = './datasets/GenImage'
    vals = ['glide', 'wukong', 'sdv4', 'midjourney', 'sdv5', 'vqdm']
    multiclasses = [0] * len(vals)
elif opt.test_dataset == 'cvpr23_ojha':
    dataroot = 'datasets/cvpr23_ojha'
    vals = ['dalle']
    multiclasses = [0] * len(vals)
elif opt.test_dataset == 'DiTFake':
    dataroot = './datasets/DiTFake/test'
    vals = ['stable-diffusion-3-medium-diffusers', 'PixArt-Sigma-XL-2-1024-MS', 'FLUX.1-schnell']
    multiclasses = [0] * len(vals)
elif opt.test_dataset == 'DRCT-2M':
    dataroot= './datasets/DRCT-2M'
    vals = ['sdv2.1', 'sdxl1.0', 'sd-turbo', 'sdxl-turbo']
    multiclasses = [0] * len(vals)
else:
    raise ValueError(f'{opt.test_dataset} not supported for now')


# Run tests

for v_id, val in enumerate(vals):
    print(f'Test on {v_id + 1}/{len(vals)}: {val}')
    opt.dataroot = '{}/{}'.format(dataroot, val)
    opt.classes = os.listdir(opt.dataroot) if multiclasses[v_id] else ['']
    print(opt.dataroot)
    print("opt.classes", opt.classes)

    opt.no_resize = True    # testing without resizing by default

    acc, ap, r_acc, f_acc, auc = validate(model, opt, opt.device)
    rows.append([val, acc, ap, r_acc, f_acc, auc])
    print("({}) acc: {}; ap: {};  r_acc: {}, f_acc: {} auc: {}".format(val, acc, ap, r_acc, f_acc, auc))


if not os.path.exists(opt.results_dir):
    os.makedirs(opt.results_dir)

csv_name = opt.results_dir + '/test_{}.csv'.format(opt.test_dataset)
with open(csv_name, 'w') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
