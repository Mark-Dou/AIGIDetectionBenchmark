import os
import sys
import torch
import argparse
import loralib as lora
import datasets

from transformers import TrainingArguments

from data.CL_dataset import create_sequential_datasets
from options.CL_train_options import CLTrainOptions

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    args = CLTrainOptions().parse()

    # Prepare sequential dataset for continual learning
    data = create_sequential_datasets(args, args.dataset_name, args.multiclass, args.dataroot)
    train_dataset = {name: data[name] for name in args.dataset_name}

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

    # Loading weights for the initial detection model
    if args.initial_weight_path is not None:

        state_dict = torch.load(args.initial_weight_path, map_location='cpu')
        print(state_dict.keys())
        try:
            if args.detection_model in ['GramNet', 'FreDect']:
                try:
                    model.load_state_dict(state_dict['netC'], strict=True)
                except:
                    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['netC'].items()},
                                          strict=True)
            elif args.detection_model == 'UnivFD':
                model.fc.load_state_dict(state_dict)
            elif args.detection_model == 'NPR':
                try:
                    # for pre-trained weights from open-source repository
                    model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict['model'].items()},
                                          strict=True)
                except:
                    # for re-implemented weights from open-source code
                    model.load_state_dict(state_dict, strict=True)
            elif args.detection_model == 'SimE':
                model.load_weights(args.initial_weight_path)
            else:
                model.load_state_dict(state_dict['model'], strict=True)
        except:
            # for re-trained models using our training script
            model.load_state_dict(state_dict['model'], strict=True)

        print('Loading checkpoints from {} successfully'.format(args.initial_weight_path))

    # Set trainable weights in the continual learning process
    if args.detection_model == 'UnivFD':
        for name, param in model.named_parameters():
            if name == "fc.weight" or name == "fc.bias":
                param.requires_grad = True
            else:
                param.requires_grad = False

    elif args.detection_model == 'SimE':
        lora.mark_only_lora_as_trainable(model)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True

    else:
        pass

    # Continual training configurations
    train_args = dict(
        model=model,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            warmup_steps=args.num_warmup_steps,
            num_train_epochs=args.num_train_epochs,
            bf16=False,
            remove_unused_columns=False,
            logging_steps=10,
            optim='adamw_torch',
            evaluation_strategy='no',
            save_strategy='no',
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            output_dir=args.output_dir,
            save_total_limit=3,
            load_best_model_at_end=True,
            ddp_find_unused_parameters=None,
            group_by_length=False,
            do_eval=False,
        ),
        opt=args,
        train_dataset=train_dataset,
        device=device
    )
    
    print(args.cl_method)
    # Continual learning method to be used
    if args.cl_method is None:
        from methods.CLBaseTrainer import CLBaseTrainer
        cl_trainer = CLBaseTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'Seq':
        # sequential training on the new task without any strategy
        from methods.Seq import SeqTrainer
        cl_trainer = SeqTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'Joint':
        # joint all historical dataset to learn
        from methods.JOINT import JointTrainer
        cl_trainer = JointTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'ER':
        # ER
        from methods.ER import ERTrainer
        cl_trainer = ERTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'EWC':
        # EWC
        from methods.EWC import EWCTrainer
        cl_trainer = EWCTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'OSLA':
        # OSLA
        from methods.OSLA import OSLATrainer
        cl_trainer = OSLATrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'AGEM':
        # AGEM
        from methods.AGEM import AveGEMTrainer
        cl_trainer = AveGEMTrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'SI':
        # SI
        from methods.SI import SITrainer
        cl_trainer = SITrainer(args.skip_initial_training, **train_args)

    elif args.cl_method == 'iCaRL':
        # iCaRL
        from methods.iCaRL import iCaRLTrainer
        cl_trainer = iCaRLTrainer(args.skip_initial_training, **train_args
                                  )
    elif args.cl_method == 'Linear':
        # Linear
        from methods.Linear import LinearTrainer
        cl_trainer = LinearTrainer(args.skip_initial_training, **train_args)
        
    else:
        ValueError(f"continual learning method: {args.cl_method} not implement yet")

    # Continual learning process
    cl_trainer.continual_learning()

if __name__ == '__main__':
    main()

