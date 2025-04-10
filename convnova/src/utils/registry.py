optimizer = {
    "adam": "torch.optim.Adam",
    "adamw": "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd": "torch.optim.SGD",
    "lamb": "src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant": "transformers.get_constant_schedule",
    "plateau": "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step": "torch.optim.lr_scheduler.StepLR",
    "multistep": "torch.optim.lr_scheduler.MultiStepLR",
    "cosine": "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup": "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup": "transformers.get_cosine_schedule_with_warmup",
    "cosine_warmup_timm": "src.utils.optim.schedulers.TimmCosineLRScheduler",
}

model = {
    # Backbones from this repo
    "model": "src.models.sequence.SequenceModel",
    "lm": "src.models.sequence.long_conv_lm.ConvLMHeadModel",
    "blm": "src.models.sequence.long_conv_lm.BertLMHeadModel",
    "lm_simple": "src.models.sequence.simple_lm.SimpleLMHeadModel",
    "vit_b_16": "src.models.baselines.vit_all.vit_base_patch16_224",
    "dna_embedding": "src.models.sequence.dna_embedding.DNAEmbeddingModel",
    "bpnet": "src.models.sequence.hyena_bpnet.HyenaBPNet",
    # "convnext": "src.models.sequence.convNext.ConvNeXt",
    "convnova": "src.models.ConvNova.convnova.CNNModel",
    # "nconvnext": "src.models.sequence.convNext.NConvNeXt",
    # "dna_bert2": "src.models.DNABERT2.DNABERT2CustomModel",
    # "caduceus": "src.models.Caduceus.caduceus.Caduceus",
    # "visualizer": "src.models.sequence.visualizer.CNNModel",
    "ntv2": "src.models.NTV2.ntv2.NTV2",
    "legnet": "src.models.LegNet.LegNet.LegNet",
    "basenji2": "src.models.basenji2.basenji2.Basenji2",
}

layer = {
    "id": "src.models.sequence.base.SequenceIdentity",
    "ff": "src.models.sequence.ff.FF",
    "mha": "src.models.sequence.mha.MultiheadAttention",
    "s4d": "src.models.sequence.ssm.s4d.S4D",
    "s4_simple": "src.models.sequence.ssm.s4_simple.SimpleS4Wrapper",
    "long-conv": "src.models.sequence.long_conv.LongConv",
    "h3": "src.models.sequence.h3.H3",
    "h3-conv": "src.models.sequence.h3_conv.H3Conv",
    "hyena": "src.models.sequence.hyena.HyenaOperator",
    "hyena-filter": "src.models.sequence.hyena.HyenaFilter",
    "vit": "src.models.sequence.mha.VitAttention",
    "ssm": "src.models.sequence.pyramid.Mamba",
    "pyramid": "src.models.sequence.mha.MultiheadAttention",
    "bert": "src.models.sequence.pyramid.BertLayer"
}

layer_config = {
    "nt": "src.models.sequence.pyramid.NucleotideTransformerConfig"
}

callbacks = {
    "timer": "src.callbacks.timer.Timer",
    "params": "src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint": "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping": "pytorch_lightning.callbacks.EarlyStopping",
    "swa": "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary": "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar": "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing": "src.callbacks.progressive_resizing.ProgressiveResizing",
    "seqlen_warmup": "src.callbacks.seqlen_warmup.SeqlenWarmup",
    "seqlen_warmup_reload": "src.callbacks.seqlen_warmup_reload.SeqlenWarmupReload",
    "gpu_affinity": "src.callbacks.gpu_affinity.GpuAffinity"
}

model_state_hook = {
    'load_backbone': 'src.models.sequence.long_conv_lm.load_backbone',
}
