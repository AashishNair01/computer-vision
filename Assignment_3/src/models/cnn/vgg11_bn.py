from torchvision.models import vgg11_bn
import torch
import torch.nn as nn

from ..base_model import BaseModel


class VGG11_bn(BaseModel):
    def __init__(self, layer_config, num_classes, activation, norm_layer, fine_tune, weights=None):
        super(VGG11_bn, self).__init__()

        ####### TODO ###########################################################
        # Initialize the different model parameters from the config file       #
        # Basically store them in `self`                                       #
        # Note that this config is for the MLP head (and not the VGG backbone) #
        ########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self.layer_config = layer_config
        self.num_classes = num_classes
        self.activation = activation
        self.norm_layer = norm_layer
        self.fine_tune = fine_tune
        self.weights = weights
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        self._build_model()

    def _build_model(self):
        #################################################################################
        # TODO: Build the classification network described in Q4 using the              #
        # models.vgg11_bn network from torchvision model zoo as the feature extraction  #
        # layers and two linear layers on top for classification. You can load the      #
        # pretrained ImageNet weights based on the weights variable. Set it to None if  #
        # you want to train from scratch, 'DEFAULT'/'IMAGENET1K_V1' if you want to use  #
        # pretrained weights. You can either write it here manually or in config file   #
        # You can enable and disable training the feature extraction layers based on    # 
        # the fine_tune flag.                                                           #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        #load the pretrained VGG11_bn model
        vgg = vgg11_bn(weights=self.weights)
        self.features = vgg.features

        #freeze the feature extraction layers if fine_tune is False
        if not self.fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False

        #Adaptive pooling
        self.avgpool = vgg.avgpool

        #Custom classifier layers
        classifier_layers = []
        input_dim = 512 * 7 * 7 
        prev_dim = input_dim

        for hidden_dim in self.layer_config:
            classifier_layers.append(nn.Linear(prev_dim, hidden_dim))
            classifier_layers.append(self.norm_layer(hidden_dim))
            classifier_layers.append(self.activation())
            prev_dim = hidden_dim

        #final output layer
        classifier_layers.append(nn.Linear(prev_dim, self.num_classes))
        self.classifier = nn.Sequential(*classifier_layers)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    def forward(self, x):
        #################################################################################
        # TODO: Implement the forward pass computation.                                 #
        # Do not apply any softmax on the output                                        #
        #################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return out
    