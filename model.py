import torch
from torch import nn

from models import densenet, resnet, mobilenet, mobilenetv2, shufflenet, shufflenetv2, squeezenet


def get_module_name(name):
    name = name.split('.')
    if name[0] == 'module':
        i = 1
    else:
        i = 0
    if name[i] == 'features':
        i += 1

    return name[i]


def get_fine_tuning_parameters(model, ft_begin_module):
    if not ft_begin_module:
        return model.parameters()

    parameters = []
    add_flag = False
    for k, v in model.named_parameters():
        if ft_begin_module == get_module_name(k):
            add_flag = True

        if add_flag:
            parameters.append({'params': v})

    return parameters


def generate_model(opt):
    assert opt.model in [
        'resnet','densenet', 'mobilenet', 'densenet', 'mobilenetv2', 'shufflenet', 'shufflenetv2', 'squeezenet'
    ]

    if opt.model == 'resnet':
        model = resnet.generate_model(model_depth=opt.model_depth,
                                      n_classes=opt.n_classes,
                                      n_input_channels=opt.n_input_channels,
                                      shortcut_type=opt.resnet_shortcut,
                                      conv1_t_size=opt.conv1_t_size,
                                      conv1_t_stride=opt.conv1_t_stride,
                                      no_max_pool=opt.no_max_pool,
                                      widen_factor=opt.resnet_widen_factor)
    # elif opt.model == 'resnet2p1d':
    #     model = resnet2p1d.generate_model(model_depth=opt.model_depth,
    #                                       n_classes=opt.n_classes,
    #                                       n_input_channels=opt.n_input_channels,
    #                                       shortcut_type=opt.resnet_shortcut,
    #                                       conv1_t_size=opt.conv1_t_size,
    #                                       conv1_t_stride=opt.conv1_t_stride,
    #                                       no_max_pool=opt.no_max_pool,
    #                                       widen_factor=opt.resnet_widen_factor)
    # elif opt.model == 'wideresnet':
    #     model = wide_resnet.generate_model(
    #         model_depth=opt.model_depth,
    #         k=opt.wide_resnet_k,
    #         n_classes=opt.n_classes,
    #         n_input_channels=opt.n_input_channels,
    #         shortcut_type=opt.resnet_shortcut,
    #         conv1_t_size=opt.conv1_t_size,
    #         conv1_t_stride=opt.conv1_t_stride,
    #         no_max_pool=opt.no_max_pool)
    # elif opt.model == 'resnext':
    #     model = resnext.generate_model(model_depth=opt.model_depth,
    #                                    cardinality=opt.resnext_cardinality,
    #                                    n_classes=opt.n_classes,
    #                                    n_input_channels=opt.n_input_channels,
    #                                    shortcut_type=opt.resnet_shortcut,
    #                                    conv1_t_size=opt.conv1_t_size,
    #                                    conv1_t_stride=opt.conv1_t_stride,
    #                                    no_max_pool=opt.no_max_pool)
    # elif opt.model == 'preresnet':
    #     model = pre_act_resnet.generate_model(
    #         model_depth=opt.model_depth,
    #         n_classes=opt.n_classes,
    #         n_input_channels=opt.n_input_channels,
    #         shortcut_type=opt.resnet_shortcut,
    #         conv1_t_size=opt.conv1_t_size,
    #         conv1_t_stride=opt.conv1_t_stride,
    #         no_max_pool=opt.no_max_pool)
    elif opt.model == 'densenet':
        model = densenet.generate_model(model_depth=opt.model_depth,
                                        n_classes=opt.n_classes,
                                        n_input_channels=opt.n_input_channels,
                                        conv1_t_size=opt.conv1_t_size,
                                        conv1_t_stride=opt.conv1_t_stride,
                                        no_max_pool=opt.no_max_pool)
    elif opt.model == 'mobilenet':
        model = mobilenet.generate_model(n_classes=opt.n_classes)
    elif opt.model == 'mobilenetv2':
        model = mobilenetv2.generate_model(n_classes=opt.n_classes)
    elif opt.model == 'shufflenet':
        model = shufflenet.generate_model(n_classes=opt.n_classes)
    elif opt.model == 'shufflenetv2':
        model = shufflenetv2.generate_model(n_classes=opt.n_classes)
    elif opt.model == 'squeezenet':
        model = squeezenet.generate_model(n_classes=opt.n_classes)

    return model


def load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes):
    if pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, map_location='cpu')

        model.load_state_dict(pretrain['state_dict'])
        tmp_model = model
        if model_name == 'resnet':
            tmp_model.fc = nn.Linear(tmp_model.fc.in_features,
                                     n_finetune_classes)
        elif model_name == 'squeezenet':
            tmp_model.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv3d(512, n_finetune_classes, kernel_size=1))
        else:
            tmp_model.classifier = nn.Linear(tmp_model.classifier.in_features,
                                             n_finetune_classes)

    return model


def make_data_parallel(model, is_distributed, device):
    if is_distributed:
        if device.type == 'cuda' and device.index is not None:
            torch.cuda.set_device(device)
            model.to(device)

            model = nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[device])
        else:
            model.to(device)
            model = nn.parallel.DistributedDataParallel(model)
    elif device.type == 'cuda':
        model = nn.DataParallel(model, device_ids=None).cuda()

    return model
