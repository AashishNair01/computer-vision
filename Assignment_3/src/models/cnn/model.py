import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch

from ..base_model import BaseModel


class ConvNet(BaseModel):
    def __init__(self, input_size, hidden_layers, num_classes, activation, norm_layer, drop_prob=0.0):
        super(ConvNet, self).__init__()

        ############## TODO ###############################################
        # Initialize the different model parameters from the config file  #
        # (basically store them in self)                                  #
        ###################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.drop_prob = drop_prob
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):

        #################################################################################
        # TODO: Initialize the modules required to implement the convolutional layer    #
        # described in the exercise.                                                    #
        # For Q1.a make use of conv2d and relu layers from the torch.nn module.         #
        # For Q2.a make use of BatchNorm2d layer from the torch.nn module.              #
        # For Q3.b Use Dropout layer from the torch.nn module if drop_prob > 0          #
        # Do NOT add any softmax layers.                                                #
        #################################################################################
        layers = []
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        input_size = self.input_size
        for each in self.hidden_layers:
            layer = nn.Conv2d(input_size, each, kernel_size=3, stride=1,padding=1)
            input_size = each
            layers.append(layer)
            layers.append(self.activation())
            layer = nn.MaxPool2d(kernel_size=2,stride=2)
            layers.append(layer)
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(self.hidden_layers[-1],self.num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def _normalize(self, img):
        """
        Helper method to be used for VisualizeFilter. 
        This is not given to be used for Forward pass! The normalization of Input for forward pass
        must be done in the transform presets.
        """
        max = np.max(img)
        min = np.min(img)
        return (img-min)/(max-min)    
    
    def VisualizeFilter(self):
        ################################################################################
        # TODO: Implement the functiont to visualize the weights in the first conv layer#
        # in the model. Visualize them as a single image fo stacked filters.            #
        # You can use matlplotlib.imshow to visualize an image in python                #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        weights = self.network[0].weight
        total = weights.shape[0]
        fig, axs = plt.subplots(8,16)
        for cur,axes in enumerate(axs.flat):
            a = weights[cur].permute(1, 2, 0).detach().numpy()
            a =self._normalize(a)
            axes.imshow(a)
            axes.axis('off')
        plt.show()


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #############################################################################
        # TODO: Implement the forward pass computations                             #
        # This can be as simple as one line :)
        # Do not apply any softmax on the logits.                                   #
        #############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****        
        x = self.network(x)
        x= torch.flatten(x,1)
        out = self.output_layer(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
