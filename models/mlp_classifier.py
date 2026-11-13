import torch.nn as nn
from configurations.configuration import (IMAGE_DIM, HIDDEN_DIM, LATENT_DIM, NUM_CLASSES)

class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        #layer 1
        self.fc1 = nn.Linear(IMAGE_DIM, HIDDEN_DIM)
        #layer 2
        self.fc2 = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        #output layer
        self.fc3 = nn.Linear(LATENT_DIM, NUM_CLASSES)
        self.relu = nn.ReLU()

    #extract feature (latent)
    def encode(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h))
        return z

    #classification
    def forward(self, x):
        x = x.view(x.size(0), -1)
        h = self.relu(self.fc1(x))
        z = self.relu(self.fc2(h))
        logits = self.fc3(z)
        return logits