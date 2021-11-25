from torch import nn
from timm import create_model

rexnet_list = ['rexnet_100', 'rexnetr_100', 'rexnet_130', 'rexnetr_130', \
                'rexnet_150', 'rexnetr_150', 'rexnet_200', 'rexnetr_200']

def get_layer_list(model_name):
    if model_name in ['efficientnet_b0', 'efficientnet_b1']:
        return [16,24,40,80,192]
    elif model_name == 'efficientnet_b2':
        return [16,24,48,88,208]
    elif model_name == 'efficientnet_b3':
        return [24,32,48,96,232]
    elif model_name == 'efficientnet_b4':
        return [24,32,56,112,272]
    elif model_name == 'efficientnet_b5':
        return [24,40,64,128,304]
    elif model_name == 'efficientnet_b6':
        return [32,40,72,144,344]
    elif model_name == 'rexnet_100':
        return [16,38,61,128,185]
    elif model_name == 'rexnetr_100':
        return [16,40,64,128,184]
    elif model_name == 'rexnet_130':
        return [21,50,79,167,240]
    elif model_name == 'rexnetr_130':
        return [24,48,80,168,240]
    elif model_name == 'rexnet_150':
        return [24,58,92,193,277]
    elif model_name == 'rexnetr_150':
        return [24,56,96,192,280]
    elif model_name == 'rexnet_200':
        return [32,77,122,257,370]
    elif model_name == 'rexnetr_200':
        return [32,80,120,256,368]

def create_encoder(args):
    model_name = args.model_parameters['unet_encoder']
    if model_name in ['efficientnet_b'+str(i) for i in range(7)]:
        encoder = _create_efficienenet_encoder(args)
    elif model_name in rexnet_list:
        encoder = _create_rexnet_encoder(args)
    return encoder

def _create_efficienenet_encoder(args):
    model = args.model_parameters['model'].lower()
    model_name = args.model_parameters['unet_encoder']
    stem_filters = args.model_parameters['stem_filters']
    squeeze_bottleneck = args.model_parameters['squeeze_bottleneck']
    squeeze_bottleneck = squeeze_bottleneck or (model == 'prob_unet')
    num_classes = args.model_parameters['num_classes']
    eff_net = create_model(model_name, in_chans=stem_filters, num_classes=num_classes)
    output_layers = eff_net.classifier.in_features
    layers_list = get_layer_list(model_name)
    enc_layers = [
        nn.Sequential(
            eff_net.conv_stem,
            eff_net.bn1,
            eff_net.act1,
            eff_net.blocks[0]
        )
    ]
    enc_layers.extend(
        [eff_net.blocks[i] for i in range(1,4)]
    )
    pre_last = [
        eff_net.blocks[4],
        eff_net.blocks[5],
        eff_net.blocks[6],
        eff_net.conv_head,
        eff_net.bn2,
        eff_net.act2
    ]
    if squeeze_bottleneck:
        enc_layers.append(
            nn.Sequential(
                eff_net.global_pool,
                eff_net.classifier
            )
        )
    else:
        pre_last.append(nn.Conv2d(output_layers, layers_list[-1], kernel_size=1))
        enc_layers.append(nn.Sequential(*pre_last))
    return nn.Sequential(*enc_layers)

def _create_rexnet_encoder(args):
    model = args.model_parameters['model'].lower()
    model_name = args.model_parameters['unet_encoder']
    stem_filters = args.model_parameters['stem_filters']
    squeeze_bottleneck = args.model_parameters['squeeze_bottleneck']
    squeeze_bottleneck = squeeze_bottleneck or (model == 'prob_unet')
    num_classes = args.model_parameters['num_classes']
    rexnet = create_model(model_name, in_chans=stem_filters, num_classes=num_classes)
    output_layers = rexnet.head.fc.in_features
    layers_list = get_layer_list(model_name)
    enc_layers = [
        nn.Sequential(
            rexnet.stem,
            rexnet.features[0]
        ),
        nn.Sequential(
            rexnet.features[1],
            rexnet.features[2]
        ),
        nn.Sequential(
            rexnet.features[3],
            rexnet.features[4]
        ),
        nn.Sequential(
            *[rexnet.features[i] for i in range(5,11)]
        ),
            [rexnet.features[i] for i in range(11,17)]
    ]
    if squeeze_bottleneck:
        enc_layers[-1] = nn.Sequential(*enc_layers[-1])
        enc_layers.append(
            nn.Sequential(
                rexnet.head.global_pool,
                rexnet.head.fc
            )
        )
    else:
        enc_layers[-1].append(nn.Conv2d(output_layers, layers_list[-1], kernel_size=1))
        enc_layers[-1] = nn.Sequential(*enc_layers[-1])
    return nn.Sequential(*enc_layers)

def add_droput(model, p):
    for name, child in model.named_children():
        if child.__class__.__name__ == 'LastConv':
            continue
        if isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear):
            old = getattr(model, name)
            new = nn.Sequential(nn.Dropout2d(p=p), old)
            setattr(model, name, new)
        else:
            add_droput(child, p)
        