import torch.nn as nn
import src.hg_utils.bert.modeling_bert as hpx_base_models


class Hyperx(hpx_base_models.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        super().__init__(hpx_base_models.BertConfig())
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict, hpx_args):
        for config in model_config_dict.values():
            config.update(vars(hpx_args))

        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        task_name, lang_name = task_name.split(':')
        _ = kwargs.pop('task_type')
        #for i in kwargs.keys():
        #    kwargs[i] = kwargs[i].cuda()
        return self.taskmodels_dict[task_name](task_name=task_name, lang_name=lang_name, **kwargs)
