import torch
import torch.nn as nn

import torch.nn.functional as F
import torch.nn.init as init

try:
    from model.RAP.modules import se_modules as se
except:
    from modules import se_modules as se

class GenericBlock(nn.Module):
    """
    Generic parent class for a conv encoder/decoder block.

    :param params: {'kernel_h': 5
                        'kernel_w': 5
                        'num_channels':64
                        'num_filters':64
                        'stride_conv':1
                        }
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(GenericBlock, self).__init__()
        if se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
        else:
            self.SELayer = None
        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)
        self.out_channel = params['num_filters']
        self.conv = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                              kernel_size=(
                                  params['kernel_h'], params['kernel_w']),
                              padding=(padding_h, padding_w),
                              stride=params['stride_conv'])
        self.prelu = nn.PReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=params['num_filters'])
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Custom weights for convolution, defaults to None
        :type weights: torch.tensor [FloatTensor], optional
        :return: [description]
        :rtype: [type]
        """

        _, c, h, w = input.shape
        if weights is None:
            x1 = self.conv(input)
        else:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(self.out_channel, c, 1, 1)
            x1 = F.conv2d(input, weights)
        x2 = self.prelu(x1)
        x3 = self.batchnorm(x2)
        return x3


class SDnetEncoderBlock(GenericBlock):
    """
    A standard conv -> prelu -> batchnorm-> maxpool block without dense connections

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(SDnetEncoderBlock, self).__init__(params, se_block_type)
        self.maxpool = nn.MaxPool2d(
            kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor]
        """

        out_block = super(SDnetEncoderBlock, self).forward(input, weights)#就是个卷积

        if self.SELayer:#没用SE
            out_block = self.SELayer(out_block, weights)
        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)#indices是把二维坐标变成一维坐标
        return out_encoder, out_block, indices


class SDnetDecoderBlock(GenericBlock):
    """Standard decoder block with maxunpool -> skipconnections -> conv -> prelu -> batchnorm, without dense connections and an optional SE blocks

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :param se_block_type: Squeeze and Excite block type to be included, defaults to None
    :type se_block_type: str, valid options are {'NONE', 'CSE', 'SSE', 'CSSE'}, optional
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params, se_block_type=None):
        super(SDnetDecoderBlock, self).__init__(params, se_block_type)
        self.unpool = nn.MaxUnpool2d(
            kernel_size=params['pool'], stride=params['stride_pool'])
        # self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
        #                       kernel_size=(1,1),
        #                       stride=params['stride_conv'])


    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward pass
        :rtype: torch.tensor
        """

        # unpool = self.unpool(input, indices) # , out_block.shape)
        unpool = F.interpolate(input, scale_factor=2, mode='nearest')
        # unpool = self.conv1(unpool)
        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(SDnetDecoderBlock, self).forward(concat, weights)
        if self.SELayer:
            out_block = self.SELayer(out_block, weights)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block

class ClassifierBlock(nn.Module):
    """
    Last layer

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_c':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'se_block': se.SELayer.None,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(
            params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights for classifier regression, defaults to None
        :type weights: torch.tensor (N), optional
        :return: logits
        :rtype: torch.tensor
        """
        batch_size, channel, a, b = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out_conv = F.conv2d(input, weights)
        else:
            out_conv = self.conv(input)
        return out_conv


class AddCoords(nn.Module):

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()
        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)#(1,y_dim,x_dim) 列向量扩张原来维度在最后
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)#(1,y_dim,x_dim)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)#(n,1,x_dim,y_dim)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)#(n,1,x_dim,y_dim)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels+2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class AF(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AF, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(1, 1, kernel_size=5, padding=2)

        self.reset_parameters()

    def forward(self, x, v):
        batch, channels, height, width = x.size()
        # x和v两个维度各增加2
        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        padded_v = F.pad(v, [self.padding, self.padding, self.padding, self.padding]) #v不用卷积
        q_out = self.query_conv(x)#(n,16,h14,h14) 卷积最后用来相乘 作为q 没有padding  而k和v都要unfold之后保持尺寸 所以才padding
        k_out = self.key_conv(padded_x)#(n,16,h16,w16) 把padding之后的来算k 是为了后面unfold
        v_out = padded_v#(n,16,h16,w16) v只是padding了 没有卷积

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)#先第2维度滑动 窗口为kernel_size 步长为stride；然后再第3维度滑动 窗口为kernel_size 步长为stride  (2,16,16,16)->(2,16,14,16,3)->(2,16,14,14,3,3)  依次按照h维度和w维度滑动
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)#(2,16,16,16)->(2,16,14,16,3)->(2,16,14,14,3,3)


        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)#第二维度(channel)16分成2个8   为什么分成h和w  这样似乎没有道理 因为通道维度并不是前面代表h 后面代表w
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)#各加个随机变量初始 (2,16,14,14,3,3)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1) # -1 = k*k  (2,1,16,14,14,9)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)#([2, 1, 16, 14, 14, 9])


        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)#([2, 1, 16, 14, 14, 1])

        out = q_out * k_out#(2,1,16,14,14,9)
        out = F.softmax(out, dim=-1)#(2,1,16,14,14,9)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)# q*k 最后 和 向量乘

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)
