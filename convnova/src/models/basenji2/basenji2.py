from .model import Basenji2 as bsj2
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class Basenji2(nn.Module):
    def __init__(self, params, d_output, seq_length, d_model=512, repeat_conv_tower=6, repeat_dilation=11, use_cropping=True, **kwargs):
        super().__init__()
        with open(params) as params_open:
            model_params = json.load(params_open)['model']
        model_params["head_human"]["units"] = d_output
        model_params["seq_length"] = seq_length
        model_params["target_length"] = seq_length
        model_params['trunk'][1]['repeat'] = repeat_conv_tower
        model_params['trunk'][2]['repeat'] = repeat_dilation
        if not use_cropping:
            model_params['trunk'].pop(3)
        # model_params['trunk'][2]['in_channels'] = int(model_params['trunk'][1]['repeat']['in_channels']**repeat_conv_tower)
        # model_params['trunk'][4]['in_channels'] = int(model_params['trunk'][1]['repeat']['in_channels']**repeat_conv_tower)
        self.d_model = d_model
        model_params['trunk'][-1]['filters'] = d_model
        model_params['head_human']['in_features'] = d_model
        

        self.basenji2 = bsj2(model_params)

    def forward(self, input_ids, position_ids=None, inference_params=None, state=None): # state for the repo interface
        if isinstance(input_ids, list):
            input_ids_tensor = input_ids[0]
            attention_mask = input_ids[1]
        else:
            input_ids_tensor = torch.tensor(input_ids)
            attention_mask = None
        if position_ids is not None:
            position_ids_tensor = position_ids
        else:
            position_ids_tensor = None
        
        x = F.one_hot(input_ids, num_classes=5).float()

        outputs = self.basenji2(
            x=x.permute(0,2,1),
            return_only_embeddings=True,
            inputs_embeds=None,
            output_hidden_states=None,
            return_dict=None,
        )
        hidden_states = outputs.permute(0,2,1)
        return hidden_states, None

    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.
        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model