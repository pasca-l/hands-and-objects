import torch.nn as nn
import torch
import numpy as np

from models.SFP.helper.stem_helper import VideoModelStem
from models.SFP.helper.fuse_helper import FuseFastToSlow
from models.SFP.helper.resnet_helper import ResStage
from models.SFP.helper.transformer_helper import Perceiver


class SlowFastPreceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_module = nn.BatchNorm3d
        self.num_pathways = 2
        self.enable_detection = True
        self._construct_network()
        self._init_weight()

    def _construct_network(self, with_head=True):
        pool_size = [[1, 1, 1], [1, 1, 1]]

        # ResNet configs
        model_stage_depth = (3, 4, 6, 3)
        d2, d3, d4, d5 = model_stage_depth
        num_groups = 1
        width_per_group = 64
        dim_inner = num_groups * width_per_group

        # SlowFast configs
        beta_inv = 8
        fusion_conv_ratio = 2
        out_dim_ratio = beta_inv // fusion_conv_ratio

        temp_kernel = [
            [[1], [5]],
            [[1], [3]],
            [[1], [3]],
            [[3], [3]],
            [[3], [3]],
        ]

        # Network
        self.s1 = VideoModelStem(
            dim_in=[3, 3],
            dim_out=[width_per_group, width_per_group // beta_inv],
            kernel=[temp_kernel[0][0] + [7, 7], temp_kernel[0][1] + [7, 7]],
            stride=[[1, 2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3, 3],
                [temp_kernel[0][1][0] // 2, 3, 3],
            ],
            norm_module=self.norm_module
        )
        self.s1_fuse = FuseFastToSlow(
            dim_in=width_per_group // beta_inv,
            fusion_conv_channel_ratio=fusion_conv_ratio,
            fusion_kernel=5,
            alpha=8,
            norm_module=self.norm_module
        )

        self.s2 = ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // beta_inv,
            ],
            dim_out=[width_per_group * 4, width_per_group * 4 // beta_inv],
            dim_inner=[dim_inner, dim_inner // beta_inv],
            temp_kernel_sizes=temp_kernel[1],
            stride=[1, 1],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[3, 3],
            nonlocal_inds=[[], []],
            nonlocal_group=[1, 1],
            nonlocal_pool=None,
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[1, 1],
            norm_module=self.norm_module
        )
        self.s2_fuse = FuseFastToSlow(
            dim_in=width_per_group * 4 // beta_inv,
            fusion_conv_channel_ratio=fusion_conv_ratio,
            fusion_kernel=5,
            alpha=4,
            norm_module=self.norm_module
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool3d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0, 0],
            )
            self.add_module(f"pathway{pathway}_pool", pool)

        self.s3 = ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // beta_inv,
            ],
            dim_out=[width_per_group * 8, width_per_group * 8 // beta_inv],
            dim_inner=[dim_inner * 2, dim_inner * 2 // beta_inv],
            temp_kernel_sizes=temp_kernel[2],
            stride=[2, 2],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[4, 4],
            nonlocal_inds=[[], []],
            nonlocal_group=[1, 1],
            nonlocal_pool=None,
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[1, 1],
            norm_module=self.norm_module
        )
        self.s3_fuse = FuseFastToSlow(
            dim_in=width_per_group * 8 // beta_inv,
            fusion_conv_channel_ratio=fusion_conv_ratio,
            fusion_kernel=5,
            alpha=4,
            norm_module=self.norm_module
        )

        self.s4 = ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // beta_inv,
            ],
            dim_out=[width_per_group * 16, width_per_group * 16 // beta_inv],
            dim_inner=[dim_inner * 4, dim_inner * 4 // beta_inv],
            temp_kernel_sizes=temp_kernel[3],
            stride=[2, 2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[6, 6],
            nonlocal_inds=[[], []],
            nonlocal_group=[1, 1],
            nonlocal_pool=None,
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[1, 1],
            norm_module=self.norm_module
        )
        self.s4_fuse = FuseFastToSlow(
            dim_in=width_per_group * 16 // beta_inv,
            fusion_conv_channel_ratio=fusion_conv_ratio,
            fusion_kernel=5,
            alpha=4,
            norm_module=self.norm_module
        )

        self.s5 = ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // beta_inv,
            ],
            dim_out=[width_per_group * 32, width_per_group * 32 // beta_inv],
            dim_inner=[dim_inner * 8, dim_inner * 8 // beta_inv],
            temp_kernel_sizes=temp_kernel[4],
            stride=[2, 2],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=[3, 3],
            nonlocal_inds=[[], []],
            nonlocal_group=[1, 1],
            nonlocal_pool=None,
            instantiation='dot_product',
            trans_func_name='bottleneck_transform',
            dilation=[1, 1],
            norm_module=self.norm_module
        )

        self.perceiver = Perceiver(
            input_channels=width_per_group + width_per_group // out_dim_ratio,
            input_axis=3,
            num_freq_bands=6,
            max_freq=5,
            depth=3,
            num_latents=256,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            attn_dropout=0.0,
            ff_dropout=0.0,
            weight_tie_layers=True,
            self_per_cross_attn=6
        )

    def forward(self, x, bboxes=None):
        # x = [x[:,:,:4,:,:], x[:,:,4:,:,:]]
        x = self.s1([torch.index_select(x, 1, torch.linspace(0, x.shape[1] - 1, x.shape[1]).long()), x])
        print(np.array(x.shape))
        x = self.s1_fuse(x)

        # use fused slow path results as input to perceiver
        x1 = x[0]  # use slow path input
        x1 = self.perceiver(x1)
        # x1 = torch.randn(x1.shape[0], 512).cuda() # debug

        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)

        # combine slow, fast, perceiver output
        x.append(x1)
        head = getattr(self, self.head_name)
        if self.enable_detection:
            x = head(x, bboxes)
        else:
            x = head(x)
        return x

    def _init_weight(self, fc_init_std=0.01, zero_init_final_bn=True):
        """
        Performs ResNet style weight initialization.
        Args:
            fc_init_std (float): the expected standard deviation for fc layer.
            zero_init_final_bn (bool): if True, zero initialize the final bn for
                every bottleneck.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                self.c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm3d):
                if (
                        hasattr(m, "transform_final_bn")
                        and m.transform_final_bn
                        and zero_init_final_bn
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def c2_msra_fill(self, module: nn.Module) -> None:
        """
        Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
        Also initializes `module.bias` to 0.
        Args:
            module (torch.nn.Module): module to initialize.
        """
        # pyre-fixme[6]: For 1st param expected `Tensor` but got `Union[Module, Tensor]`.
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            # pyre-fixme[6]: Expected `Tensor` for 1st param but got `Union[nn.Module,
            #  torch.Tensor]`.
            nn.init.constant_(module.bias, 0)