import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import itertools

class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters
    are fixed
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x):
        scale = self.weight * self.running_var.rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class _TransNorm(Module):

    """http: // ise.thss.tsinghua.edu.cn / ~mlong / doc / transferable - normalization - nips19.pdf"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super(_TransNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features))
            self.bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        if self.track_running_stats:
            self.register_buffer('running_mean_source', torch.zeros(num_features))
            self.register_buffer('running_mean_target', torch.zeros(num_features))
            self.register_buffer('running_var_source', torch.ones(num_features))
            self.register_buffer('running_var_target', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean_source', None)
            self.register_parameter('running_mean_target', None)
            self.register_parameter('running_var_source', None)
            self.register_parameter('running_var_target', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.track_running_stats:
            self.running_mean_source.zero_()
            self.running_mean_target.zero_()
            self.running_var_source.fill_(1)
            self.running_var_target.fill_(1)
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def _check_input_dim(self, input):
        return NotImplemented

    def _load_from_state_dict_from_pretrained_model(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        r"""Copies parameters and buffers from :attr:`state_dict` into only
        this module, but not its descendants. This is called on every submodule
        in :meth:`~torch.nn.Module.load_state_dict`. Metadata saved for this
        module in input :attr:`state_dict` is provided as :attr`metadata`.
        For state dicts without meta data, :attr`metadata` is empty.
        Subclasses can achieve class-specific backward compatible loading using
        the version number at `metadata.get("version", None)`.
        .. note::
            :attr:`state_dict` is not the same object as the input
            :attr:`state_dict` to :meth:`~torch.nn.Module.load_state_dict`. So
            it can be modified.
        Arguments:
            state_dict (dict): a dict containing parameters and
                persistent buffers.
            prefix (str): the prefix for parameters and buffers used in this
                module
            metadata (dict): a dict containing the metadata for this moodule.
                See
            strict (bool): whether to strictly enforce that the keys in
                :attr:`state_dict` with :attr:`prefix` match the names of
                parameters and buffers in this module
            missing_keys (list of str): if ``strict=False``, add missing keys to
                this list
            unexpected_keys (list of str): if ``strict=False``, add unexpected
                keys to this list
            error_msgs (list of str): error messages should be added to this
                list, and will be reported together in
                :meth:`~torch.nn.Module.load_state_dict`
        """
        local_name_params = itertools.chain(self._parameters.items(), self._buffers.items())
        local_state = {k: v.data for k, v in local_name_params if v is not None}

        for name, param in local_state.items():
            key = prefix + name
            # if 'source' in key or 'target' in key:
            #     key = key[:-7]
            #     print(key)
            if key in state_dict:
                input_param = state_dict[key]
                if input_param.shape != param.shape:
                    # local shape should match the one in checkpoint
                    error_msgs.append('size mismatch for {}: copying a param of {} from checkpoint, '
                                      'where the shape is {} in current model.'
                                      .format(key, param.shape, input_param.shape))
                    continue
                if isinstance(input_param, Parameter):
                    # backwards compatibility for serialized parameters
                    input_param = input_param.data
                try:
                    param.copy_(input_param)
                except Exception:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}.'
                                      .format(key, param.size(), input_param.size()))
            elif strict:
                missing_keys.append(key)



    def forward(self, input):
        self._check_input_dim(input)
        if self.training :  ## train mode

            ## 1. Domain Specific Mean and Variance.
            batch_size = input.size()[0] // 2
            input_source = input[:batch_size]
            input_target = input[batch_size:]

            ## 2. Domain Sharing Gamma and Beta.
            z_source = F.batch_norm(
                input_source, self.running_mean_source, self.running_var_source, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)

            z_target = F.batch_norm(
                input_target, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)
            z = torch.cat((z_source, z_target), dim=0)

            if input.dim() == 4:    ## TransNorm2d
                input_source = input_source.permute(0,2,3,1).contiguous().view(-1,self.num_features)
                input_target = input_target.permute(0,2,3,1).contiguous().view(-1,self.num_features)

            cur_mean_source = torch.mean(input_source, dim=0)
            cur_var_source = torch.var(input_source,dim=0)
            cur_mean_target = torch.mean(input_target, dim=0)
            cur_var_target = torch.var(input_target, dim=0)

            ## 3. Domain Adaptive Alpha.

            ### 3.1 Calculating Distance
            dis = torch.abs(cur_mean_source / torch.sqrt(cur_var_source + self.eps) -
                            cur_mean_target / torch.sqrt(cur_var_target + self.eps))

            ### 3.2 Generating Probability
            prob = 1.0 / (1.0 + dis)
            alpha = self.num_features * prob / sum(prob)

            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)

            ## 3.3 Residual Connection
            return z * (1 + alpha.detach())


        else:  ##test mode
            z = F.batch_norm(
                input, self.running_mean_target, self.running_var_target, self.weight, self.bias,
                self.training or not self.track_running_stats, self.momentum, self.eps)

            dis = torch.abs(self.running_mean_source / torch.sqrt(self.running_var_source + self.eps)
                               - self.running_mean_target / torch.sqrt(self.running_var_target + self.eps))
            prob = 1.0 / (1.0 + dis)
            alpha = self.num_features * prob / sum(prob)

            if input.dim() == 2:
                alpha = alpha.view(1, self.num_features)
            elif input.dim() == 4:
                alpha = alpha.view(1, self.num_features, 1, 1)
            return z * (1 + alpha.detach())

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = metadata.get('version', None)
        if (version is None or version < 2) and self.track_running_stats:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        self._load_from_state_dict_from_pretrained_model(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

class TransNorm2d(_TransNorm):
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class conv_2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, activation='relu', norm_layer=None):
        super(conv_2d, self).__init__()
        if activation == 'relu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.ReLU(inplace=True)
            )
        elif activation == 'tanh':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.Tanh()
            )
        elif activation == 'leakyrelu':
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
                norm_layer(out_ch),
                nn.LeakyReLU()
            )


    def forward(self, x):
        x = self.conv(x)
        return x


class fc_layer(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
        super(fc_layer, self).__init__()
        if activation == 'relu':
            self.ac = nn.ReLU(inplace=True)
        elif activation == 'leakyrelu':
            self.ac = nn.LeakyReLU()
        if bn:
            bnlayer = nn.BatchNorm1d(out_ch)
            for param in bnlayer.parameters():
                param.requires_grad = True
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                bnlayer,
                self.ac
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(in_ch, out_ch),
                self.ac
            )

    def forward(self, x):
        x = self.fc(x)
        return x


class transform_net(nn.Module):
    def __init__(self, in_ch, K=3, device="cpu", norm_layer=None):
        super(transform_net, self).__init__()    
        self.K = K
        self.conv2d1 = conv_2d(in_ch, 64, 1, norm_layer=norm_layer)
        self.conv2d2 = conv_2d(64, 128, 1, norm_layer=norm_layer)
        self.conv2d3 = conv_2d(128, 1024, 1, norm_layer=norm_layer)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
        self.fc1 = fc_layer(1024, 512)
        self.fc2 = fc_layer(512, 256)
        self.fc3 = nn.Linear(256, K*K)
        self.device = device

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.conv2d2(x)
        x = self.conv2d3(x)
        x, _ = torch.max(x, dim=2, keepdim=False)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1).to(self.device)
        x = x + iden
        x = x.view(x.size(0), self.K, self.K)
        return x

class Pointnet_encoder(nn.Module):
    def __init__(self, device="cpu", feat_dims=1024, freeze_bn=False):
        super(Pointnet_encoder, self).__init__()
        if freeze_bn:
            norm_layer = FrozenBatchNorm2d
        else:
            norm_layer = nn.BatchNorm2d
            # norm_layer = nn.InstanceNorm2d
            # norm_layer = TransNorm2d

        self.trans_net1 = transform_net(3,3, device, norm_layer=norm_layer)
        self.trans_net2 = transform_net(64,64, device, norm_layer=norm_layer)
        self.conv1 = conv_2d(3, 64, 1, norm_layer=norm_layer)
        self.conv2 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv3 = conv_2d(64, 64, 1, norm_layer=norm_layer)
        self.conv4 = conv_2d(64, 128, 1, norm_layer=norm_layer)
        self.conv5 = conv_2d(128, feat_dims, 1, norm_layer=norm_layer)
        self.device = device
    
    def forward(self, x):
            x = x.permute(0, 2, 1).unsqueeze(dim=3) #B, C, N, 1 
            transform = self.trans_net1(x)
            x = x.transpose(2, 1)
            x = x.squeeze()
            x = torch.bmm(x, transform)
            x = x.unsqueeze(3)
            x = x.transpose(2, 1)
            x = self.conv1(x)
            x = self.conv2(x)
            transform = self.trans_net2(x)
            x = x.transpose(2, 1)
            x = x.squeeze()
            x = torch.bmm(x, transform)
            x = x.unsqueeze(3)
            x = x.transpose(2, 1)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            feature, _ = torch.max(x, dim=2, keepdim=False)
            x = feature.squeeze()#batchsize*1024
            return x

class Pointnet_cls(nn.Module):
    def __init__(self, num_class=10, feat_dims=512):
        super(Pointnet_cls, self).__init__()
        self.mlp1 = fc_layer(feat_dims, 512)
        self.dropout1 = nn.Dropout2d(p=0.5)
        self.mlp2 = fc_layer(512, 256)
        self.dropout2 = nn.Dropout2d(p=0.5)
        self.mlp3 = nn.Linear(256, num_class)
    
    def forward(self, x):
        x = self.mlp1(x)#batchsize*512
        x = self.dropout1(x)
        x = self.mlp2(x)#batchsize*256
        x = self.dropout2(x)
        x = self.mlp3(x)#batchsize*10
        return x


class Pointnet(nn.Module):
    def __init__(self, num_class=10, device="cpu", feat_dims=512, freeze_bn=False, tgt_decoder=False):
        super(Pointnet, self).__init__()
        self.encoder = Pointnet_encoder(device, feat_dims, freeze_bn)
        self.decoder = Pointnet_cls(num_class=num_class, feat_dims=feat_dims)
        if tgt_decoder:
            self.decoder_target = Pointnet_cls(num_class=num_class, feat_dims=feat_dims)

    
    def forward(self, x, embeddings=False, from_features=False, target=False):
        if from_features:
            if target:
                x = self.decoder_target(x)
            else:
                x = self.decoder(x)
            return x
        else:
            feature = self.encoder(x)
            if target:
                x = self.decoder_target(feature)
            else:
                x = self.decoder(feature)
            if embeddings:
                return feature, x
            else:
                return x

# import torch
# import torch.nn as nn

# class FrozenBatchNorm2d(nn.Module):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters
#     are fixed
#     """

#     def __init__(self, n):
#         super(FrozenBatchNorm2d, self).__init__()
#         self.register_buffer("weight", torch.ones(n))
#         self.register_buffer("bias", torch.zeros(n))
#         self.register_buffer("running_mean", torch.zeros(n))
#         self.register_buffer("running_var", torch.ones(n))

#     def forward(self, x):
#         scale = self.weight * self.running_var.rsqrt()
#         bias = self.bias - self.running_mean * scale
#         scale = scale.reshape(1, -1, 1, 1)
#         bias = bias.reshape(1, -1, 1, 1)
#         return x * scale + bias


# class conv_2d(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel, activation='relu', norm_layer=None):
#         super(conv_2d, self).__init__()
#         if activation == 'relu':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
#                 norm_layer(out_ch),
#                 nn.ReLU(inplace=True)
#             )
#         elif activation == 'tanh':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
#                 norm_layer(out_ch),
#                 nn.Tanh()
#             )
#         elif activation == 'leakyrelu':
#             self.conv = nn.Sequential(
#                 nn.Conv2d(in_ch, out_ch, kernel_size=kernel),
#                 norm_layer(out_ch),
#                 nn.LeakyReLU()
#             )


#     def forward(self, x):
#         x = self.conv(x)
#         return x


# class fc_layer(nn.Module):
#     def __init__(self, in_ch, out_ch, bn=True, activation='leakyrelu'):
#         super(fc_layer, self).__init__()
#         if activation == 'relu':
#             self.ac = nn.ReLU(inplace=True)
#         elif activation == 'leakyrelu':
#             self.ac = nn.LeakyReLU()
#         if bn:
#             bnlayer = nn.BatchNorm1d(out_ch)
#             for param in bnlayer.parameters():
#                 param.requires_grad = True
#             self.fc = nn.Sequential(
#                 nn.Linear(in_ch, out_ch),
#                 bnlayer,
#                 self.ac
#             )
#         else:
#             self.fc = nn.Sequential(
#                 nn.Linear(in_ch, out_ch),
#                 self.ac
#             )

#     def forward(self, x):
#         x = self.fc(x)
#         return x


# class transform_net(nn.Module):
#     def __init__(self, in_ch, K=3, device="cpu", norm_layer=None):
#         super(transform_net, self).__init__()    
#         self.K = K
#         self.conv2d1 = conv_2d(in_ch, 64, 1, norm_layer=norm_layer)
#         self.conv2d2 = conv_2d(64, 128, 1, norm_layer=norm_layer)
#         self.conv2d3 = conv_2d(128, 1024, 1, norm_layer=norm_layer)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(512, 1))
#         self.fc1 = fc_layer(1024, 512)
#         self.fc2 = fc_layer(512, 256)
#         self.fc3 = nn.Linear(256, K*K)
#         self.device = device

#     def forward(self, x):
#         x = self.conv2d1(x)
#         x = self.conv2d2(x)
#         x = self.conv2d3(x)
#         x, _ = torch.max(x, dim=2, keepdim=False)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.fc3(x)

#         iden = torch.eye(self.K).view(1,self.K * self. K).repeat(x.size(0),1).to(self.device)
#         x = x + iden
#         x = x.view(x.size(0), self.K, self.K)
#         return x


# class Pointnet(nn.Module):
#     def __init__(self, num_class=10, device="cpu", feat_dims=512, freeze_bn=False):
#         super(Pointnet, self).__init__()
#         if freeze_bn:
#             norm_layer = FrozenBatchNorm2d
#         else:
#             norm_layer = nn.BatchNorm2d

#         self.trans_net1 = transform_net(3,3, device, norm_layer=norm_layer)
#         self.trans_net2 = transform_net(64,64, device, norm_layer=norm_layer)
#         self.conv1 = conv_2d(3, 64, 1, norm_layer=norm_layer)
#         self.conv2 = conv_2d(64, 64, 1, norm_layer=norm_layer)
#         self.conv3 = conv_2d(64, 64, 1, norm_layer=norm_layer)
#         self.conv4 = conv_2d(64, 128, 1, norm_layer=norm_layer)
#         self.conv5 = conv_2d(128, feat_dims, 1, norm_layer=norm_layer)

#         self.mlp1 = fc_layer(feat_dims, 512)
#         self.dropout1 = nn.Dropout2d(p=0.5)
#         self.mlp2 = fc_layer(512, 256)

#         self.dropout2 = nn.Dropout2d(p=0.5)
#         self.mlp3 = nn.Linear(256, num_class)
#         self.device = device

#     def forward(self, x, embeddings=False, from_features=False):
#         # batch_size = x.size(0)
#         # point_num = x.size(2)

#         # x = self.farthest_point_sample(x.permute(0, 2, 1), hcfg("pc_input_num")) #B, C, N
#         if from_features:
#             x = self.mlp1(x)#batchsize*512
#             x = self.dropout1(x)
#             x = self.mlp2(x)#batchsize*256
#             x = self.dropout2(x)
#             x = self.mlp3(x)#batchsize*10
#             return x

#         else:
#             x = x.permute(0, 2, 1).unsqueeze(dim=3) #B, C, N, 1 
            
#             transform = self.trans_net1(x)
#             x = x.transpose(2, 1)
#             x = x.squeeze()
#             x = torch.bmm(x, transform)
#             x = x.unsqueeze(3)
#             x = x.transpose(2, 1)
#             x = self.conv1(x)
#             x = self.conv2(x)
#             transform = self.trans_net2(x)
#             x = x.transpose(2, 1)
#             x = x.squeeze()
#             x = torch.bmm(x, transform)
#             x = x.unsqueeze(3)
#             x = x.transpose(2, 1)
#             x = self.conv3(x)
#             x = self.conv4(x)
#             x = self.conv5(x)
#             feature, _ = torch.max(x, dim=2, keepdim=False)
#             x = feature.squeeze()#batchsize*1024

#             x = self.mlp1(x)#batchsize*512
#             x = self.dropout1(x)
#             x = self.mlp2(x)#batchsize*256
#             x = self.dropout2(x)
#             x = self.mlp3(x)#batchsize*10

#             if embeddings:
#                 return feature, x
#             else:
#                 return x
