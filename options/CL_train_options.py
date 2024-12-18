from .base_options import BaseOptions

def list_of_strings(arg):
    return arg.split(',')

class CLTrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # dataset configuration
        parser.add_argument('--dataset_name', type=list_of_strings, help='Sequential datasets to be used in continual learning process.')
        parser.add_argument('--multiclass', nargs="+", type=int, default=[1], help="Whether task have mutilclass dataset")

        # AI-generated detection model
        parser.add_argument('--initial_weight_path', type=str, default='', help='Weight path for initial detector')

        # Continual learning method for AI-generated image detection
        parser.add_argument('--cl_method', default=None, choices=['Seq', 'Joint', 'ER', 'EWC', 'OSLA', 'iCaRL', 'SI', 'AGEM', 'Linear'], help='Continual learning method to be used')

        parser.add_argument('--skip_initial_training', nargs="+", type=bool, default=False, help="Whether task0 have been done")
        parser.add_argument('--output_hidden_states', nargs="+", type=bool, default=False, help="whether output_hidden_states")
        parser.add_argument('--return_lora_features', nargs="+", type=bool, default=False, help="whether return_lora_features")
        parser.add_argument('--return_features_logits', nargs="+", type=bool, default=False, help="whether return_features_logits")

        # Configuration in the continual learning process
        parser.add_argument("--num_train_epochs", type=int, default=5, help="in the paper 'TRACE: A Comprehensive Benchmark for Continual Learning in Large Language Models http://arxiv.org/abs/2310.06762', better set the epochs to 5 ")
        parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader.")
        parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size (per device) for the evaluation dataloader.")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.")
        parser.add_argument("--weight_decay", type=float, default=0., help="Weight decay to use.")
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.", )
        parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
        parser.add_argument("--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler.")
        parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Ratio of total training steps used for warmup.")
        parser.add_argument("--output_dir", type=str, default=None, help="Where to store the model.")
        parser.add_argument('--eval_steps', default=500, help='eval steps')
        parser.add_argument('--save_steps', default=2000, help='save steps')
        parser.add_argument('--threshold', type=float, default=0.90, help='threshold on whether continual learning')

        self.isTrain = True
        
        return parser
