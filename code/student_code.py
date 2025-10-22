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
# Part I.1: Understanding Convolutions
################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation.
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

        # Get dimensions
        batch_size = input_feats.size(0)
        in_channels = input_feats.size(1)
        out_channels = weight.size(0)
        
        # Use unfold to create sliding windows
        # unfold creates a tensor of shape (N, C*K*K, L) where L is number of windows
        input_unfold = unfold(
            input_feats, 
            kernel_size=(kernel_size, kernel_size),
            padding=padding, 
            stride=stride
        )
        
        # Reshape weight for matrix multiplication
        # From (Co, Ci, K, K) to (Co, Ci*K*K)
        weight_flat = weight.view(out_channels, -1)
        
        # Perform convolution as matrix multiplication
        # (Co, Ci*K*K) @ (N, Ci*K*K, L) -> (N, Co, L)
        output = weight_flat @ input_unfold
        
        # Add bias if present
        if bias is not None:
            output = output + bias.view(1, -1, 1)
        
        # Calculate output spatial dimensions
        output_height = (input_feats.size(2) + 2 * padding - kernel_size) // stride + 1
        output_width = (input_feats.size(3) + 2 * padding - kernel_size) // stride + 1
        
        # Reshape output to (N, Co, H_out, W_out)
        output = output.view(batch_size, out_channels, output_height, output_width)
        
        # Save for backward
        ctx.save_for_backward(input_unfold, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward propagation of convolution operation
        """
        # unpack tensors and initialize the grads
        input_unfold, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width
        
        batch_size = grad_output.size(0)
        out_channels = grad_output.size(1)
        in_channels = weight.size(1)

        # Reshape grad_output for computation
        # (N, Co, H_out, W_out) -> (N, Co, L)
        grad_output_flat = grad_output.view(batch_size, out_channels, -1)

        # Compute gradient w.r.t. weight
        if ctx.needs_input_grad[1]:
            # grad_weight: (Co, Ci*K*K) = grad_output_flat @ input_unfold^T
            # (N, Co, L) @ (N, L, Ci*K*K) -> sum over batch -> (Co, Ci*K*K)
            grad_weight = grad_output_flat @ input_unfold.transpose(1, 2)
            grad_weight = grad_weight.sum(dim=0)  # Sum over batch
            # Reshape to (Co, Ci, K, K)
            grad_weight = grad_weight.view(out_channels, in_channels, kernel_size, kernel_size)

        # Compute gradient w.r.t. input
        if ctx.needs_input_grad[0]:
            # grad_input_unfold: (N, Ci*K*K, L) = weight^T @ grad_output_flat
            # (Ci*K*K, Co) @ (N, Co, L) -> (N, Ci*K*K, L)
            weight_flat = weight.view(out_channels, -1)
            grad_input_unfold = weight_flat.transpose(0, 1) @ grad_output_flat
            
            # Use fold to convert back to image format
            grad_input = fold(
                grad_input_unfold,
                output_size=(input_height, input_width),
                kernel_size=(kernel_size, kernel_size),
                padding=padding,
                stride=stride
            )

        # Compute gradient w.r.t. bias
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
    """Improved CNN with Batch Normalization and Residual Connections"""
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual block 1
        self.res_block1 = self._make_residual_block(conv_op, 64, 128, stride=1)
        
        # Residual block 2
        self.res_block2 = self._make_residual_block(conv_op, 128, 256, stride=2)
        
        # Residual block 3
        self.res_block3 = self._make_residual_block(conv_op, 256, 512, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
        # For adversarial training
        self.adversarial_training = False
        self.attacker = None

    def _make_residual_block(self, conv_op, in_channels, out_channels, stride=1):
        """Create a residual block with skip connection"""
        layers = []
        
        # Main path
        layers.append(conv_op(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(conv_op(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        
        main_path = nn.Sequential(*layers)
        
        # Skip connection
        if stride != 1 or in_channels != out_channels:
            skip_connection = nn.Sequential(
                conv_op(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )
        else:
            skip_connection = nn.Identity()
        
        return ResidualBlock(main_path, skip_connection)

    def forward(self, x):
        # Adversarial training during training phase
        if self.training and self.adversarial_training and self.attacker is not None:
            # Generate adversarial samples
            with torch.no_grad():
                adv_x = self.attacker.perturb(self, x.detach())
            # Mix original and adversarial samples
            batch_size = x.size(0)
            indices = torch.randperm(batch_size)
            mixed_x = torch.cat([x[indices[:batch_size//2]], adv_x[indices[batch_size//2:]]], dim=0)
            x = mixed_x
        
        x = self.conv1(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def enable_adversarial_training(self, attacker):
        """Enable adversarial training with given attacker"""
        self.adversarial_training = True
        self.attacker = attacker


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, main_path, skip_connection):
        super(ResidualBlock, self).__init__()
        self.main_path = main_path
        self.skip_connection = skip_connection
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = self.skip_connection(x)
        out = self.main_path(x)
        out += identity
        out = self.relu(out)
        return out


# Change this to your improved model
default_cnn_model = SimpleNet


################################################################################
# Part II.1: Understanding self-attention and Transformer block
################################################################################
class Attention(nn.Module):
    """Multi-head Self-Attention."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # linear projection for query, key, value
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # linear projection at the end
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
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
        
        # Compute attention scores: Q * K^T / sqrt(d_k)
        # (B * nHead, H * W, C) @ (B * nHead, C, H * W) -> (B * nHead, H * W, H * W)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply softmax to get attention weights
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        # (B * nHead, H * W, H * W) @ (B * nHead, H * W, C) -> (B * nHead, H * W, C)
        x = attn @ v
        
        # Reshape back to (B, H, W, C)
        x = x.reshape(B, self.num_heads, H * W, -1).transpose(1, 2)
        x = x.reshape(B, H, W, -1)
        
        # Final linear projection
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer blocks with support of local window self-attention"""
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
            act_layer=act_layer
        )

        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)

        # Local window attention if window_size > 0
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            # Partition into windows
            x, pad_hw = window_partition(x, self.window_size)
            # Apply attention within windows
            x = self.attn(x)
            # Merge windows back
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        else:
            # Global attention
            x = self.attn(x)

        # First residual connection with stochastic depth
        x = shortcut + self.drop_path(x)
        
        # MLP block with second residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


#################################################################################
# Part II.2: Design and train a vision Transformer
#################################################################################
class SimpleViT(nn.Module):
    """
    Vision Transformer with local window attention
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
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding
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

        # Create Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            # Use window attention for specified blocks, global attention otherwise
            block_window_size = window_size if i in window_block_indexes else 0
            
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=block_window_size,
            )
            self.blocks.append(block)
        
        # Final layer norm
        self.norm = norm_layer(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize weights
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add positional embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        
        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.mean(dim=[1, 2])
        
        # Classification head
        x = self.head(x)
        
        return x


# Change this to your model
default_vit_model = SimpleViT


# Data augmentation for training
def get_train_transforms():
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    ))
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms


# Data augmentation for validation
def get_val_transforms():
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
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
        Project Gradient Descent attack
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Generate adversarial samples using PGD
        """
        # Clone input and enable gradients
        output = input.clone().detach()
        output.requires_grad = True
        
        # Store original input for projection
        original_input = input.clone().detach()
        
        # Set model to eval mode for attack
        was_training = model.training
        model.eval()
        
        # PGD iterations
        for step in range(self.num_steps):
            # Zero gradients
            if output.grad is not None:
                output.grad.zero_()
            
            # Forward pass
            predictions = model(output)
            
            # Get least confident label (target for attack)
            # We want to minimize confidence in the correct prediction
            target = predictions.argmin(dim=1)
            
            # Compute loss for the least confident label
            loss = self.loss_fn(predictions, target)
            
            # Backward pass
            loss.backward()
            
            # FGSM step: sign of gradient
            with torch.no_grad():
                # Gradient ascent (we want to maximize loss)
                perturbation = self.step_size * output.grad.sign()
                output = output + perturbation
                
                # Project back to epsilon ball around original input
                delta = output - original_input
                delta = torch.clamp(delta, -self.epsilon, self.epsilon)
                output = original_input + delta
                
                # Clamp to valid image range [0, 1] after normalization
                # Note: input is already normalized, so we need to be careful
                output = torch.clamp(output, 
                                   input.min().item(), 
                                   input.max().item())
                
                # Detach and require grad for next iteration
                output = output.detach()
                output.requires_grad = True
        
        # Restore model training state
        if was_training:
            model.train()
        
        return output.detach()


default_attack = PGDAttack


def vis_grid(input, n_rows=10):
    """
    Visualize a batch of images as a grid
    """
    # concat all images into a big picture
    output_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    return output_imgs


default_visfunction = vis_grid