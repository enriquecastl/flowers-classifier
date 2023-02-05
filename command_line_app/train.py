from workspace_utils import active_session
from nn_utils import choose_device
from model import Model
from datasets import Datasets
from terminal import get_train_arguments
        
def main():
    args = get_train_arguments()
    device = choose_device(args.gpu)
    model = Model(device, args.arch, args.hidden_units, args.learning_rate)
    datasets = Datasets(args.data_dir)
    epochs = args.epochs
    
    print(f"Training network using {device} device with architecture {args.arch}")
    print(f"Hyperparameters:\nHidden units: {args.hidden_units}\nEpochs: {args.epochs}\nLearning rate: {args.learning_rate}")
       
    with active_session():
        for epoch in range(epochs):
            running_loss = model.train(datasets.training_loader)
            validation_loss, accuracy = model.eval(datasets.validation_loader)

            print(f"Epoch {epoch + 1}/{epochs})..\n"
                  f"Train loss: {running_loss:.3f}..\n"
                  f"Validation loss: {validation_loss:.3f}..\n"
                  f"Validation accuracy: {accuracy:.3f}..")
    
        
        print("Starting testing phase")
        
        validation_loss, accuracy = model.eval(datasets.testing_loader)
        
        print(f"Running the model on the testing dataset yield an accuracy of {accuracy:.3%} and loss of {validation_loss:.3f}")

        model.save_checkpoint(args.save_dir, datasets.training_dataset)
        
if __name__ == "__main__":
    main()