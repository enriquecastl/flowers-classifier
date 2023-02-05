import argparse

def get_train_arguments():
    parser = argparse.ArgumentParser(description="A program that trains a neural network to classify flower images")
    
    parser.add_argument("--arch", help="CNN model architecture", default="vgg16", choices=['vgg16', 'densenet121'])
    parser.add_argument("--learning_rate", help="CNN learning rate", default="0.003", type=float)
    parser.add_argument("--hidden_units", help="CNN classifier hidden layer units", default="512", type=int)
    parser.add_argument("--epochs", help="Training epochs", default="10", type=int)
    parser.add_argument("--save_dir", help="Where to save the models checkpoints", default="saved_models")
    parser.add_argument("--gpu", help="Use gpu to run the model", action='store_true')
    parser.add_argument('data_dir', metavar="data-directory")    
    
    return parser.parse_args()


def get_predict_arguments():
    parser = argparse.ArgumentParser(description="A program that uses a neural network to classify flowers")
    
    parser.add_argument("--top_k", help="Top K most likely flowers", default=3, type=int)
    parser.add_argument("--category_names", help="A file that maps categories to real names", default=None)
    parser.add_argument("--gpu", help="Use gpu to run the model", action='store_true')
    parser.add_argument("image", help="Image to classify")
    parser.add_argument("checkpoint", help="Trained model checkpoint to use in predictions")

    return parser.parse_args()
