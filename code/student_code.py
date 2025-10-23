import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torchvision.utils import make_grid
import math

from utils import resize_image
import custom_transforms as transforms
from custom_blocks import (PatchEmbed, window_partition, window_unpartition,
                           DropPath, MLP, trunc_normal_)


################################################################################
# You will need to fill in the missing code in this file
################################################################################


################################################################################
# Part I.1: Understanding Convolutions
################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation. We only consider square
        filters with equal stride and padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input.
            Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        ########################################################################
        # Fill in the code here

        # Input dimensions (batch size, # input channels, input height, input width)
        # Ex. batch of 10 RGB images sized 32×32: input_feats.shape == (10, 3, 32, 32)
        N, C_in, H, W = input_feats.shape 
        
        # Weight dimensions (# filters, # input channels, filter kernel size)
        # Ex. 16 filters of size 3×3 on RGB input: weight.shape == (16, 3, 3, 3)
        C_out, _, K, _ = weight.shape 

        # Compute output feature map spatial dimensions
        H_out = (H + 2 * padding - K) // stride + 1
        W_out = (W + 2 * padding - K) // stride + 1

        # Unfold input into shape (N, (C_in * K * K), (H_out * W_out)):
        # the torch unfold() function takes care of:
        #   applying zero padding, 
        #   extracting all sliding patches of size (C_in, K, K),
        #   spacing them correctly according to the stride,
        #   stacking all column vectors side by side
        input_unf = torch.nn.functional.unfold(input_feats, kernel_size=K, padding=padding, stride=stride)

        # Matrix multiplication:
        # weights
        weight_flat = weight.view(C_out, -1) # each row = 1 flattened filter
        output_unf = weight_flat  @ input_unf  # (N, C_out, (H_out * W_out))

        # bias
        if bias is not None:
            output_unf += bias.view(1, -1, 1)

        # Reshape to output
        output = output_unf.view(N, C_out, H_out, W_out).clone()

        ########################################################################

        # save for backward (you need to save the unfolded tensor into ctx)
        ctx.save_for_backward(input_unf, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs (∂L/∂Y; where Y = Conv(X, W, b))

        Outputs:
          grad_input: gradients of the input features (∂L/∂X)
          grad_weight: gradients of the convolution weight (∂L/∂W)
          grad_bias: gradients of the bias term (∂L/∂b)

        """
        # unpack tensors and initialize the grads
        input_unf, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width

        ########################################################################
        # Fill in the code here

        # Same shape as output from forward
        N, C_out, H_out, W_out = grad_output.shape

        # Same shape as convolution filters
        C_out, C_in, K, _ = weight.shape

        # Flatten spatial dimensions so each feature map becomes a 2D matrix
        grad_output_reshaped = grad_output.reshape(N, C_out, -1)

        # Compute gradient w.r.t. input (∂L/∂X)
        # From the forward: Y_unf = W_flat * X_unf
        # So the gradient w.r.t. the unfolded input is: ∂L/∂X_unf = W_flat.t() @ ∂L/∂Y_unf
        if ctx.needs_input_grad[0]:
            grad_input_unf = weight.view(C_out, -1).t() @ grad_output_reshaped
            # then fold patches back into 2D spatial layout to get gradients w.r.t original input tensor
            grad_input = torch.nn.functional.fold(
                grad_input_unf,
                output_size=(input_height, input_width),
                kernel_size=K,
                padding=padding,
                stride=stride,
            )

        # Compute gradient w.r.t. weights (∂L/∂W)
        # From the forward: Y_unf = W_flat * X_unf
        # So the gradient w.r.t. the weights is: ∂L/∂W_flat = ∂L/∂Y_unf @ X_unf.t()
        if ctx.needs_input_grad[1]:
            # Each output channel’s gradient is a weighted sum of the input patches that contributed to it.
            grad_weight = grad_output_reshaped @ input_unf.transpose(1, 2)
            # We have one such gradient per sample in the batch, so we sum over the batch dimension:
            grad_weight = grad_weight.sum(0)
            # Finally, reshape back to the original kernel shape
            grad_weight = grad_weight.view(C_out, C_in, K, K) # (C_out, C_in, K, K) is same as weight

        ########################################################################
        # compute the gradients w.r.t. input and params

        # Compute gradient w.r.t. bias (∂L/∂b)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(
            input, self.weight, self.bias, self.stride, self.padding
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


################################################################################
# Part I.2: Design and train a convolutional network
################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if you implement adversarial training, label and configure the attack params here
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class CustomNet(nn.Module):
    # a custom CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(CustomNet, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(

            # BLOCK1: 2 Conv with Skip - Conv Kernel 7x7
            # Skip Connection 
            SkipBlock(3, 64, kernel_size=7, padding=3),

            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # BLOCK 2: 2 Conv with Skip - Conv Kernel 3x3
            SkipBlock(64, 256, kernel_size=3, padding=1),

            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # BLOCK 3: 2 Conv with Skip - Conv Kernel 3x3
            SkipBlock(256, 512, kernel_size=3, padding=1),  
        )

        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.consintat_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class CustomNet2(nn.Module):
    # a custom CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(CustomNet2, self).__init__()
        # you can start from here and create a better model
        self.features = nn.Sequential(

            # BLOCK1: Start with large conv Kernel 7x7 to get general features
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # BLOCK2: 2x 2 Conv with Skip - Conv Kernel 3x3
            SkipBlock(64, 64, kernel_size=3, padding=1),
            SkipBlock(64,64, kernel_size=3, padding=1),

            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # BLOCK 3: 2 Conv with Skip - Conv Kernel 3x3
            SkipBlock(64, 256, kernel_size=3, padding=1),

            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # BLOCK 4: 2 Conv with Skip - Conv Kernel 3x3
            SkipBlock(256, 512, kernel_size=3, padding=1),  
        )

        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        # if self.training:
        #   # generate adversarial sample based on x
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class  SkipBlock(nn.Module):
    def __init__(self, in_size , out_size, kernel_size, padding):
        super().__init__()

        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=kernel_size, stride=1, padding=padding)
        
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=kernel_size, stride=1, padding=padding)

        if (in_size == out_size):
            self.skip = nn.Identity() 
        else:
            self.skip = nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.skip(x)
        out = self.relu(out)
        return out 


# change this to your model!
default_cnn_model = CustomNet2


################################################################################
# Part II: Vision Transformer
################################################################################
class Attention(nn.Module):
    """Multi-head Attention block with relative position embeddings."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Args:
            x (tensor): input features with shape of (B, H, W, C)
        """
       # input size (B, H, W, C)
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = (
            self.qkv(x).reshape(
                B, H * W, 3, self.num_heads, -1
            ).permute(2, 0, 3, 1, 4)
        )
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        ########################################################################
        # Fill in the code here
        
        # Transpose k for matrix multiplication: (B*nHead, C, H*W)
        k_t = k.transpose(-2, -1)
        
        # Compute attention scores: (B*nHead, H*W, H*W)
        attn = (q @ k_t) * self.scale
        
        # Apply softmax to get attention weights
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values: (B*nHead, H*W, C)
        x = attn @ v
        
        # Reshape back: (B, num_heads, H*W, C) -> (B, H*W, num_heads, C)
        x = x.reshape(B, self.num_heads, H * W, -1).transpose(1, 2)
        
        # Concatenate heads: (B, H*W, num_heads*C) -> (B, H, W, dim)
        x = x.reshape(B, H, W, -1)
        
        # Final projection
        x = self.proj(x)
        ########################################################################
        return x


class TransformerBlock(nn.Module):
    """
    Transformer blocks with support of window attention and residual propagation blocks
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        window_size=0,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            window_size (int): Window size for window attention blocks.
                If it equals 0, then not use window attention.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
        )

        self.window_size = window_size

    def forward(self, x):
        """
        Args:
            x (tensor): input features with shape of (B, H, W, C)
        """
        shortcut = x
        x = self.norm1(x)
        
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        
        # Attention
        x = self.attn(x)
        
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in <https://arxiv.org/abs/2010.11929>.
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and
                can be coupled with standard global attention. E.g., [0, 2]
                indicates the first and the third blocks will use local window
                attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # patch embedding layer
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        ########################################################################
        # Fill in the code here
        ########################################################################
        # The implementation shall define some Transformer blocks

        ########################################################################
        # Create Transformer blocks
        ########################################################################
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0,
            )
            for i in range(depth)
        ])

        # Final normalization layer
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        # Forward pass through Vision Transformer
        ########################################################################
        
        # Patch embedding: (B, C, H, W) -> (B, H', W', embed_dim)
        x = self.patch_embed(x)
        
        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Global average pooling: (B, H', W', C) -> (B, C)
        x = x.mean(dim=[1, 2])
        
        # Classification head: (B, C) -> (B, num_classes)
        x = self.head(x)
        
        return x

# change this to your model!
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    train_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms

# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms():
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    # mean / std from imagenet
    val_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


################################################################################
# Part III: Adversarial samples
################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=10, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False
        original_input = input.clone()

        # loop over the number of steps
        for step in range(self.num_steps):
            output = output.detach()
            output.requires_grad = True
            
            logits = model(output)
            
            with torch.no_grad():
                least_confident_label = logits.argmin(dim=1)
            
            loss = self.loss_fn(logits, least_confident_label)
            
            model.zero_grad()
            if output.grad is not None:
                output.grad.zero_()
            loss.backward()
            
            with torch.no_grad():
                grad_sign = output.grad.sign()
                output = output + self.step_size * grad_sign
                
                perturbation = output - original_input
                perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                output = original_input + perturbation
                
                output = torch.clamp(output, -3.0, 3.0)

        return output.detach()


default_attack = PGDAttack


################################################################################
# Part III BONUS: Adversarial Training (+2 Points)
################################################################################
class SimpleNetAdversarial(nn.Module):
    """
    SimpleNet with adversarial training capability for BONUS section.
    This trains a model that is robust to adversarial attacks.
    """
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNetAdversarial, self).__init__()
        
        # Same architecture as SimpleNet
        self.features = nn.Sequential(
            # conv1 block: conv 7x7
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv2 block: simple bottleneck
            conv_op(64, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(64, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # conv3 block: simple bottleneck
            conv_op(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            conv_op(128, 512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Adversarial training configuration
        self.adversarial_training = True  # Enable adversarial training
        self.pgd_steps = 7  # Fewer steps for faster training
        self.pgd_alpha = 0.007  # Step size
        self.pgd_epsilon = 0.031  # Perturbation bound

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, targets=None):
        """
        Forward pass with optional adversarial training.
        
        Args:
            x: Input images (B, C, H, W)
            targets: True labels (B,) - required for adversarial training
        
        Returns:
            logits: Model predictions (B, num_classes)
        """
        # Generate adversarial examples during training if enabled
        if self.adversarial_training and self.training and targets is not None:
            x = self._generate_adversarial(x, targets)
        
        # Standard forward pass
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def _generate_adversarial(self, x, targets):
        """
        Generate adversarial examples using PGD during training.
        
        Args:
            x: Clean input images
            targets: True labels
            
        Returns:
            x_adv: Adversarial examples
        """
        x_orig = x.detach()
        x_adv = x.clone().detach()
        criterion = nn.CrossEntropyLoss()
        
        for _ in range(self.pgd_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            features = self.features(x_adv)
            pooled = self.avgpool(features)
            flattened = pooled.view(pooled.size(0), -1)
            logits = self.fc(flattened)
            
            # Compute loss
            loss = criterion(logits, targets)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, x_adv)[0]
            
            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv + self.pgd_alpha * grad.sign()
                
                # Project to epsilon ball
                perturbation = torch.clamp(x_adv - x_orig, -self.pgd_epsilon, self.pgd_epsilon)
                x_adv = x_orig + perturbation
                
                # Clamp to valid range
                x_adv = torch.clamp(x_adv, -3.0, 3.0)
        
        return x_adv.detach()


# IMPORTANT: Uncomment the line below ONLY when doing adversarial training (BONUS)
# This will train a robust model instead of the regular SimpleNet
# default_cnn_model = SimpleNetAdversarial


def vis_grid(input, n_rows=10):
    """
    Given a batch of image X (torch tensor), compose a mosaic for visualziation.

    Args:
      input: (torch tensor) input image of size N * C * H * W
      n_rows: (int) number of images per row

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    output_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    return output_imgs


default_visfunction = vis_grid
