import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from constants import categories_count, checkpoint_filename
import os as os
import os.path as path

class Model:
    def __init__(self, device="cpu", arch="vgg16", hidden_units=512, learning_rate=0.003, dropout=0.2):
        self.arch = arch
        self.hidden_units = hidden_units
        self.dropout = dropout
        self.model = None
        self.features_count = 0
        self.device = device
        self.learning_rate = learning_rate
        self.categories_count = categories_count
        self.criterion = nn.NLLLoss()
        
        self._download_arch()
        self._create_classifier()

        self.model.to(device)
        
    @staticmethod
    def from_checkpoint(checkpoint_filename, device='cpu'):
        checkpoint = torch.load(checkpoint_filename)
        
        model = Model(device, checkpoint['arch'], checkpoint['hidden_units'], checkpoint['learning_rate'])

        model.classifier.load_state_dict(checkpoint['state_dict'])
        model.classifier.class_to_idx = checkpoint['class_to_idx']

        return model
    
    
    def save_checkpoint(self, save_dir, training_dataset):
        if not path.isdir(save_dir):
            os.mkdir(save_dir)

        full_filename = path.join(save_dir, f"{self.arch}_{checkpoint_filename}")

        checkpoint = {
            'arch': self.arch,
            'hidden_units': self.hidden_units,
            'learning_rate': self.learning_rate,
            'state_dict': self.classifier.state_dict(),
            'class_to_idx': training_dataset.class_to_idx,
        }

        torch.save(checkpoint, full_filename)
    
    @property
    def classifier(self):
        return self.model.classifier 
    
    def train(self, data_loader):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        device = self.device
        running_loss = 0
        
        model.train()

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(data_loader) 
    
    
    def eval(self, data_loader):
        model = self.model
        criterion = self.criterion
        device = self.device
        validation_loss = 0
        accuracy = 0

        model.eval()

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model.forward(inputs)
                loss = criterion(logits, labels)

                validation_loss += loss.item()
                accuracy += self._calc_accuracy(logits, labels)

        return validation_loss / len(data_loader), accuracy / len(data_loader)
    
    
    def predict(self, torch_img, topk=5):
        model = self.model
        device = self.device
        
        model.eval()

        with torch.no_grad():
            ps = torch.exp(model.forward(torch_img.unsqueeze_(0).float().to(device)))

            return ps.topk(topk, dim=1)
    
    def _download_arch(self):
        arch = self.arch
        model = None
        features_count = 1

        if arch == 'vgg16':
            model = models.vgg16(pretrained=True)
            features_count = 25088
        elif arch == 'densenet121':
            features_count = 1024
            model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad= False

        self.model = model
        self.features_count = features_count
    
    def _create_classifier(self):
        classifier = nn.Sequential(nn.Linear(self.features_count, self.hidden_units),
                                   nn.ReLU(),
                                   nn.Dropout(self.dropout),
                                   nn.Linear(self.hidden_units, categories_count),
                                   nn.LogSoftmax(dim=1))
        self.model.classifier = classifier
        self.optimizer = optim.Adam(classifier.parameters(), lr=self.learning_rate)
            
        
    def _calc_accuracy(self, output, labels):
        ps = torch.exp(output)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)

        return torch.mean(equals.type(torch.FloatTensor)).item()
