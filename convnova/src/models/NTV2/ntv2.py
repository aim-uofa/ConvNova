from .modeling_esm import EsmForSequenceClassification
from torch import nn
from .esm_config import EsmConfig
from peft import get_peft_model, LoraConfig, TaskType
from omegaconf import OmegaConf

class NTV2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        
        # Convert DictConfig to dictionary
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Create EsmConfig
        esm_config = EsmConfig.from_dict(config_dict)
        
        # Load the pretrained model
        self.esm = EsmForSequenceClassification.from_pretrained(config_dict["_name_or_path"], config=esm_config, **kwargs)
        
        # Initialize LoRA configuration
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # or whatever task type you're using
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "query",
                "key",
                "value",
                "dense"
            ]
        )
        
        # Apply LoRA to the model
        self.esm = get_peft_model(self.esm, lora_config)
        
        self.d_model = esm_config.hidden_size

    def forward(self, input_ids, position_ids=None, inference_params=None):
        outputs = self.esm(
            input_ids,
            position_ids=position_ids,
        )
        return outputs[0].logits, None
        
    @property
    def d_output(self):
        """Model /embedding dimension, used for decoder mapping.

        """
        if getattr(self, "d_model", None) is None:
            raise NotImplementedError("SequenceModule instantiation must set d_output")
        return self.d_model