def freeze_params(module):
    for param in module.parameters():
        param.requires_grad = False

def unfreeze_params(module):
    for param in module.parameters():
        param.requires_grad = True