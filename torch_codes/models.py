import torch
import torch.nn as nn
import torch.nn.functional as F

class MDN(nn.Module):


    def __init__(self, input_dim, output_dim, n_gaussians, hidden_dim=64):
        """
        mixture density network (MDN) for modeling the conditional distribution of thetas given data.
        this implementation has a single hidden layer with ReLU activation.

        Args:
            :input_dim: int, dimension of the input data
            :output_dim: int, dimension of the output data
            :n_gaussians: int, number of Gaussian components
            :hidden_dim: int, dimension of the hidden layer

        """

        super(MDN, self).__init__()

        self.hidden = nn.Linear(input_dim, hidden_dim)  # Hidden layer
        self.pi = nn.Linear(hidden_dim, n_gaussians)    # Mixing coefficients
        self.sigma = nn.Linear(hidden_dim, n_gaussians * output_dim) # Standard deviations
        self.mu = nn.Linear(hidden_dim, n_gaussians * output_dim)    # Means

        self.n_gaussians = n_gaussians
        self.output_dim = output_dim

    def forward(self, y):
        h = F.relu(self.hidden(y))  # Hidden layer with ReLU activation
        
        pi = F.softmax(self.pi(h), dim=1)  # Mixing coefficients
        sigma = torch.exp(self.sigma(h)).view(-1, self.n_gaussians, self.output_dim)  # Standard deviations (must be positive)
        mu = self.mu(h).view(-1, self.n_gaussians, self.output_dim)  # Means

        return pi, sigma, mu


def mdn_loss(pi, sigma, mu, x):
    """
    Compute the negative log likelihood of the data given the MDN model.

    Args:
        :pi: torch.Tensor, mixing coefficients of the Gaussian components
        :sigma: torch.Tensor, standard deviations of the Gaussian components
        :mu: torch.Tensor, means of the Gaussian components
        :x: torch.Tensor, target data

    Returns:
        :nll: torch.Tensor, mean negative log likelihood of the data given the MDN model
    """
    # Expand x to match the shape of mu and sigma
    x = x.unsqueeze(1).expand_as(mu)  # (batch_size, n_gaussians, output_dim)

    # Compute Gaussian probability for each mixture component
    normal_dist = torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * torch.pi)))

    # Compute the product of probabilities across dimensions (since x is multi-dimensional)
    gaussian_prob = torch.prod(normal_dist, dim=2)  # (batch_size, n_gaussians)

    # Weight the Gaussian probabilities by the mixing coefficients
    weighted_prob = pi * gaussian_prob

    # Sum across the mixture components to compute the total probability
    total_prob = torch.sum(weighted_prob, dim=1)  # (batch_size)

    # Compute the negative log likelihood
    nll = -torch.log(total_prob + 1e-8)  # Add small constant to avoid log(0)
    return torch.mean(nll)


class FCN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        '''
        fully connected network (FCN) as a simple classifier for CE loss

        Args:
            :input_size: int, size of the input feature vector
            :hidden_layers: list of int, sizes of the hidden layers
            :output_size: int, size of the output feature vector
        
        '''
        super(FCN, self).__init__()
        
        # Create a list of fully connected layers
        layers = []
        
        # Input layer
        prev_layer_size = input_size
        for hidden_layer_size in hidden_layers:
            layers.append(nn.Linear(prev_layer_size, hidden_layer_size))
            layers.append(nn.ReLU())
            prev_layer_size = hidden_layer_size
        
        # Output layer
        layers.append(nn.Linear(prev_layer_size, output_size))
        
        # Combine all layers into a Sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        x = self.network(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, output_size):
        '''
        Simple CNN model for extracting features from the input data. Designed for 64x64 images.

        Args:
            :output_size: int, size of the output feature vector

        '''

        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, output_size) 

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.pool(F.relu(self.conv3(x)))  
        x = x.view(-1, 128 * 8 * 8) 
        x = F.relu(self.fc1(x))  
        x = self.fc2(x) 
        return x
    

