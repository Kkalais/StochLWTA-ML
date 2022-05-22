from maml import ModelAgnosticMetaLearningModel
from maml_umtra_networks import SimpleModel
from maml_benchmarks import OmniglotDatabase

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default="MAML", help='algorithm to run')
parser.add_argument('--learning_rate', type=float, default=0.001, help='meta_learning_rate')
parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='continue training from last checkpoint')
parser.add_argument('--lwta', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='use of lwta model')
parser.add_argument('--bma', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='use of bayesian model averaging')

args = parser.parse_args()

def run_omniglot():
    omniglot_database = OmniglotDatabase(
        random_seed=47,
        num_train_classes=1200,
        num_val_classes=100,
    )
    
    maml = ModelAgnosticMetaLearningModel(
        database=omniglot_database,
        network_cls=SimpleModel,
        n=5, #the N for => k-shot N-way
        k_ml=1, 
        k_val_ml=5, #the k for => k-shot N-way
        k_val=1,
        k_val_val=15,
        k_test=1,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.4,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=float(args.learning_rate),  
        report_validation_frequency=50,
        log_train_images_after_iteration=200,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='omniglot',
        val_seed=42,
        val_test_batch_norm_momentum=0.0
    )
    
    maml.train(iterations=5000, algorithm=args.algorithm, load_model=args.load_model) 
    maml.evaluate(iterations=50, num_tasks=1000, use_val_batch_statistics=True, seed=42)

if __name__ == '__main__':
    run_omniglot()
