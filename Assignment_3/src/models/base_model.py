import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod # To be implemented by child classes.
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """

        ret_str = super().__str__()
    
        #### TODO #######################################
        # Print the number of **trainable** parameters  #
        # by appending them to ret_str                  #
        #################################################        
        num_of_params = 0
        for layer_name, param in self.named_parameters():
            if param.requires_grad:
                num_of_params_in_layer = param.numel()
                num_of_params += num_of_params_in_layer
                ret_str += f"Layer name: {layer_name} has {num_of_params_in_layer} number of learnable parameters.\n"
        ret_str += f"Total number of trainable parameters in the network is {num_of_params}"
                
        
        return ret_str