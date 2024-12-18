from .base_options import BaseOptions



class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        def list_of_strings(arg):
            return arg.split(',')

        # dataset configuration
        parser.add_argument('--model_dir', type=str, help='Continual learning checkpoint dir')
        parser.add_argument('--dataset_name', type=list_of_strings, help='Dataset to be used.')
        parser.add_argument('--multiclass', nargs="+", type=int, default=[1], help="Whether task have mutilclass dataset")
        
        # continual learning method
        parser.add_argument('--cl_method', default=None, choices=['Seq', 'Joint', 'ER', 'EWC', 'OSLA', 'iCaRL', 'SI', 'AGEM'], help='Continual learning method to be used')
        parser.add_argument('--checkpoint_name', type=list_of_strings, help='Checkpoint name used in the continual learning process.')

        # save evaluation results 
        parser.add_argument('--output_dir', type=str, help='Save path for evaluation results.')

        self.isTrain = False

        return parser
