import torch
from hesiod import hcfg


def get_optimizer(network):

    if hcfg("optimizers.optimizer_name") == "sgd":
        opt = torch.optim.SGD(network, lr=hcfg("lr"), 
                            momentum=hcfg("optimizers.momentum"), weight_decay=hcfg("optimizers.weight_decay"))
    else:
        opt = torch.optim.AdamW(network, lr=hcfg("lr"), weight_decay=hcfg("optimizers.weight_decay"))
    return opt