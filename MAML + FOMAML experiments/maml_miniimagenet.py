from maml import ModelAgnosticMetaLearningModel
from maml_umtra_networks import MiniImagenetModel
from maml_benchmarks import MiniImagenetDatabase
from cdml_challenge_benchmarks import ChestXRay8Database

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default="MAML", help='algorithm to run')
parser.add_argument('--learning_rate', type=float, default=0.001, help='meta_learning_rate')
parser.add_argument('--load_model', type=lambda x: (str(x).lower() == 'true'), default=False,
                    help='continue training from last checkpoint')
args = parser.parse_args()


def run_mini_imagenet():
    mini_imagenet_database = MiniImagenetDatabase()

    maml = ModelAgnosticMetaLearningModel(
        database=mini_imagenet_database,
        #target_database=ChestXRay8Database(),
        network_cls=MiniImagenetModel,
        n=5, #the N for => k-shot N-way
        k_ml=1, 
        k_val_ml=1, #the k for => k-shot N-way
        k_val=1,
        k_val_val=15,
        k_test=15,
        k_val_test=15,
        meta_batch_size=4,
        num_steps_ml=5,
        lr_inner_ml=0.05,
        num_steps_validation=5,
        save_after_iterations=1000,
        meta_learning_rate=float(args.learning_rate),
        report_validation_frequency=100,
        log_train_images_after_iteration=1000,
        num_tasks_val=100,
        clip_gradients=True,
        experiment_name='mini_imagenet',
        val_seed=42,
        val_test_batch_norm_momentum=0.0,
    )

    maml.train(iterations=50000, algorithm=args.algorithm, load_model=args.load_model) 
    maml.evaluate(50, num_tasks=1000, seed=14, use_val_batch_statistics=True)


if __name__ == '__main__':
    run_mini_imagenet()
