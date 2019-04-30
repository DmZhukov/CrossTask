import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--primary_path',
        type=str,
        default='crosstask_release/tasks_primary.txt',
        help='list of primary tasks')
    parser.add_argument(
        '--related_path',
        type=str,
        default='crosstask_release/tasks_related.txt',
        help='list of related tasks')
    parser.add_argument(
        '--annotation_path',
        type=str,
        default='crosstask_release/annotations',
        help='path to annotations')
    parser.add_argument(
        '--video_csv_path',
        type=str,
        default='crosstask_release/videos.csv',
        help='path to video csv')
    parser.add_argument(
        '--val_csv_path',
        type=str,
        default='crosstask_release/videos_val.csv',
        help='path to validation csv')
    parser.add_argument(
        '--features_path',
        type=str,
        default='crosstask_features',
        help='path to features')
    parser.add_argument(
        '--constraints_path',
        type=str,
        default='crosstask_constraints',
        help='path to constraints')
    parser.add_argument(
        '--n_train',
        type=int,
        default=30,
        help='videos per task for training')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5,
        help='learning rate')
    parser.add_argument(
        '-q',
        type=float,
        default=0.7,
        help='regularization parameter')
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='number of training epochs')
    parser.add_argument(
        '--pretrain_epochs',
        type=int,
        default=30,
        help='number of pre-training epochs')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of dataloader workers'
        )
    parser.add_argument(
        '--use_related',
        type=int,
        default=1,
        help='1 for using related tasks during training, 0 for using primary tasks only'
        )
    parser.add_argument(
        '--use_gpu',
        type=int,
        default=0,
        )
    parser.add_argument(
        '-d',
        type=int,
        default=3200,
        help='dimension of feature vector',
        )
    parser.add_argument(
        '--lambd',
        type=float,
        default=1e4,
        help='penalty coefficient for temporal cosntraints. Put 0 to use no temporal constraints during training',
        )
    parser.add_argument(
        '--share',
        type=str,
        default='words',
        help='Level of sharing between tasks',
        )
    args = parser.parse_args()
    return args