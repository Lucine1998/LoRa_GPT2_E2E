import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoConfig

class LoRALinear(nn.Module):
    def __init__(self, weight, bias, lora_dim):
        super(LoRALinear, self).__init__()
        row, column = weight.shape

        if bias is None:
            self.linear = nn.Linear(column, row, bias=False)
            self.linear.load_state_dict({"weight": weight})
        else:
            self.linear = nn.Linear(column, row)
            self.linear.load_state_dict({"weight": weight, "bias": bias})

        self.lora_right = nn.Parameter(torch.zeros(column, lora_dim))
        nn.init.kaiming_uniform_(self.lora_right, a=math.sqrt(5))
        self.lora_left = nn.Parameter(torch.zeros(lora_dim, row))

    def forward(self, input):
        x = self.linear(input)
        y = input @ self.lora_right @ self.lora_left
        return x + y

def replace_with_lora(model, lora_dim, device):
    target_names = [name for name, module in model.named_modules() if "attn.c_attn" in name]

    for name in target_names:
        name_struct = name.split(".")
        module_list = [model]
        for struct in name_struct:
            module_list.append(getattr(module_list[-1], struct))

        lora = LoRALinear(
            weight=torch.transpose(module_list[-1].weight, 0, 1),
            bias=module_list[-1].bias,
            lora_dim=lora_dim
        ).to(device)

        module_list[-2].__setattr__(name_struct[-1], lora)

def get_model(model_name, lora_dim, device):
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, config=config).to(device)
    replace_with_lora(model, lora_dim, device)
    return model
