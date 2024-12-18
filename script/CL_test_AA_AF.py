import argparse
import os
import torch
import csv

from script.validate import validate
from options.CL_test_options import TestOptions

import warnings
warnings.filterwarnings('ignore')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(args):

    # AI-generated image detection model
    if args.detection_model == 'LGrad':
        from models.modules.LGrad.resnet import resnet50
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 1)
    elif args.detection_model == 'LNP':
        from models.modules.LNP.resnet import resnet50
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 1)
    elif args.detection_model == 'GramNet':
        from models.modules.GramNet.resnet_gram import resnet18
        model = resnet18(num_classes=1)
    elif args.detection_model == 'FreDect':
        import torchvision
        model = torchvision.models.resnet50()
        model.fc = torch.nn.Linear(2048, 1)
    elif args.detection_model == 'CNNSpot':
        from models.modules.CNNSpot.resnet50 import resnet50
        model = resnet50(pretrained=True)
        model.fc = torch.nn.Linear(2048, 1)
    elif args.detection_model == 'UnivFD':
        from models.modules.UnivFD import get_model
        model = get_model(args.arch)
    elif args.detection_model == 'NPR':
        from models.modules.NPR.resnet import resnet50
        model = resnet50(pretrained=False, num_classes=1)
    elif args.detection_model == 'SAFE':
        from models.modules.SAFE.resnet import resnet50
        model = resnet50(num_classes=2)
    elif args.detection_model == 'SimE':
        from models.modules.SimE import load_model
        model, _ = load_model(args.arch, device='cpu')
    else:
        raise ValueError(f'{args.detection_model} not supported for now')

    model_dir = os.path.join(args.model_dir, args.detection_model, args.cl_method)
    print(model_dir)

    dataroot = args.dataroot

    results = []
    acc_matrix = []
    num = 1
    for name in args.checkpoint_name:

        model_path = os.path.join(model_dir, 'checkpoint_{}.pth'.format(name))
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        print(model_path + ' loaded')

        rows = [["Path: {}".format(model_path)],
                ['TestSet', 'ACC', 'AP', 'AUC']]

        acc_array = []

        for v_id, val_dataset in enumerate(args.dataset_name):
            print(val_dataset)
            args.dataroot = '{}/{}'.format(dataroot, val_dataset)
            args.classes = os.listdir(args.dataroot) if args.multiclass[v_id] else ['']
    
            acc, ap, r_acc, f_acc, auc = validate(model, args, device)

            print([val_dataset, acc, ap, auc])
            rows.append([val_dataset, acc, ap, auc])

            acc_array.append(acc)

        results.append(rows)
        acc_matrix.append(acc_array)

        AA = 0
        for i in range(num):
            AA += acc_matrix[num - 1][i]
        AA /= num

        AF = 0
        if num > 1:
            for i in range(num - 1):
                BWT = 0
                for j in range(i + 1, num):
                    BWT += acc_matrix[j][i] - acc_matrix[i][i]
                BWT /= num - i - 1
                AF += BWT
            AF /= num - 1
            print("AA: {}, AF: {}".format(AA * 100, AF * 100))
            results.append(["AA: {}".format(AA * 100), "AF: {}".format(AF * 100)])

        else:
            print("AA: {}".format(AA * 100))
            results.append(["AA: {}".format(AA * 100)])

        num += 1

    csv_dir = os.path.join(args.output_dir, args.detection_model, args.cl_method)
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir, exist_ok=True)

    csv_name = csv_dir + '/AA_AF_results.csv'

    with open(csv_name, 'w') as f:
        csv_writer = csv.writer(f, delimiter=',')
        csv_writer.writerows(results)


if __name__ == '__main__':
    args = TestOptions().parse(print_options=False)
    main(args)