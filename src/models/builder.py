
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from run_configs import RunConfigs, ModelType, ModelVariant

from models.swinvit.swin_transformer_v2 import SwinTransformerV2
from models.reconstruction_heads.base import BaseReconstructionHead
from models.reconstruction_heads.climax import ClimaXReconstructionHead
import hiera
from vision_xformer import ViL
import models.msvit as msvit
from models.autoencoder.MAE import *


def get_params_for_model_variant(model_variant: ModelVariant):
    if model_variant == ModelVariant.PARAMS_100M:
        return 138, [2, 2, 18, 2], [6, 12, 24, 48]
    elif model_variant == ModelVariant.PARAMS_600M:
        return 270, [2, 2, 18, 2], [6, 12, 24, 48]
    elif model_variant == ModelVariant.PARAMS_1B:
        return 320, [2, 2, 42, 4], [10, 20, 40, 80]
    elif model_variant == ModelVariant.PARAMS_3B:
        return 528, [2, 2, 42, 4], [16, 32, 64, 128]
    else:
        raise ValueError("Unknown model variant")


def build_model(configs : RunConfigs): 

    if configs.model_type == ModelType.SwinTransformerV2:

        embed_dim, depths, num_heads = get_params_for_model_variant(configs.model_variant)
        backbone = SwinTransformerV2(
            in_chans=6, 
            num_classes=0,
            embed_dim=embed_dim,
            window_size=128, 
            depths=depths, 
            num_heads=num_heads, 
            input_resolution=(128, 128), 
            img_size=128,
            sequential_self_attention=False, 
            use_checkpoint=False, 
        )

        head = BaseReconstructionHead(backbone.num_features, 6*128*128, (6, 128, 128))
        # head = ClimaXReconstructionHead(backbone.num_features, 6*128*128, (6, 128, 128), embed_dim=4096, decoder_depth=2)
        backbone.head = head
        return backbone

    elif configs.model_type == ModelType.HieraTiny:
        backbone = hiera.Hiera(
            input_size = (128, 128),
            in_chans = 6,
            # embed_dim = 96,  # initial embed dim
            # num_heads = 1,  # initial number of heads
            num_classes = 10,
            stages = (1, 2, 7, 2)
            # q_pool: int = 3,  # number of q_pool stages
            # q_stride: Tuple[int, ...] = (2, 2),
            # mask_unit_size: Tuple[int, ...] = (8, 8),  # must divide q_stride ** (#stages-1)
            # mask_unit_attn: which stages use mask unit attention?
            # mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
            # dim_mul: float = 2.0,
            # head_mul: float = 2.0,
            # patch_kernel: Tuple[int, ...] = (7, 7),
            # patch_stride: Tuple[int, ...] = (4, 4),
            # patch_padding: Tuple[int, ...] = (3, 3),
            # mlp_ratio: float = 4.0,
            # drop_path_rate: float = 0.0,
            # norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
            # head_dropout: float = 0.0,
            # head_init_scale: float = 0.001,
            # sep_pos_embed: bool = False,
        )

        head = BaseReconstructionHead(768, 6*128*128, (6, 128, 128))
        backbone.head = head
        return backbone
    
    elif configs.model_type == ModelType.Linformer:
        backbone = msvit.MsViT(
            'l1,h4,d192,n1,s1,g1,p16,f7,a1_l2,h6,d384,n10,s0,g1,p2,f7,a1_l3,h16,d768,n1,s0,g1,p2,f7,a1', # THIS IS FOR LINFORMER
            img_size=128, 
            in_chans=6,
            num_classes=10,
            attn_type='linformer',
        )

        head = BaseReconstructionHead(backbone.out_planes, 6*128*128, (6, 128, 128))
        backbone.head = head
        return backbone
    
    elif configs.model_type == ModelType.Performer:
        backbone = msvit.MsViT(
            'l1,h4,d192,n1,s1,g1,p16,f7,a1_l2,h6,d384,n10,s0,g1,p2,f7,a1_l3,h16,d768,n1,s0,g1,p2,f7,a1', # THIS IS FOR PERFORMER
            img_size=128, 
            in_chans=6,
            num_classes=10,
            attn_type='performer',
        )

        head = BaseReconstructionHead(backbone.out_planes, 6*128*128, (6, 128, 128))
        backbone.head = head
        return backbone
    
    elif configs.model_type == ModelType.LongFormer:
        backbone = msvit.MsViT(
            'l1,h4,d192,n1,s1,g1,p16,f7,a1_l2,h6,d384,n10,s0,g1,p2,f7,a1_l3,h16,d768,n1,s0,g1,p2,f7,a1', # THIS IS FOR LONGFORMER
            img_size=128, 
            in_chans=6,
            num_classes=10,
            # attn_type='longformerhand'
            attn_type='longformerauto', 
        )

        head = BaseReconstructionHead(backbone.out_planes, 6*128*128, (6, 128, 128))
        backbone.head = head
        return backbone
    
    elif configs.model_type == ModelType.SRFormer:
        backbone = msvit.MsViT(
            'l1,h4,d192,n1,s1,g1,p16,f2,a1_l2,h6,d384,n10,s0,g1,p2,f2,a1_l3,h16,d768,n1,s0,g1,p2,f2,a1', # THIS IS FOR SRFORMER
            img_size=128, 
            in_chans=6,
            num_classes=10,
            attn_type='srformer',
        )

        head = BaseReconstructionHead(backbone.out_planes, 6*128*128, (6, 128, 128))
        backbone.head = head
        return backbone
    
    elif configs.model_type == ModelType.Linformer:
        backbone = ViL(
            image_size=128, 
            patch_size=16, 
            num_classes=10, 
            dim=384, 
            depth=6, 
            heads=6, 
            mlp_dim=384, 
            channels=6, 
            emb_dropout=0.1, 
            dropout=0.1, 
            attn_dropout=0.1, 
            attn_type='linformer',
        )

        head = BaseReconstructionHead(backbone.dim, 6*128*128, (6, 128, 128))
        backbone.mlp_head = head
        return backbone
    
    elif configs.model_type == ModelType.MaskedAutoencoder:
        if configs.model_variant == ModelVariant.PARAMS_100M:
            return mae_vit_base_patch16_dec512d8b()
        elif configs.model_variant == ModelVariant.PARAMS_600M:
            return mae_vit_huge_patch14_dec512d8b()
        elif configs.model_variant == ModelVariant.PARAMS_1B:
            return mae_vit_huge_custom_1B()
        elif configs.model_variant == ModelVariant.PARAMS_3B:
            return mae_vit_huge_custom_3B()
        else:
            raise ValueError("Unknown model variant")

    else:
        raise ValueError("Unknown model type")
    
