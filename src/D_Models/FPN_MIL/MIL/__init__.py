import torch 
import math
from torch import nn

# internal imports 
from src.D_Models.FPN_MIL.FeatureExtractors import Define_Feature_Extractor, FeaturePyramidNetwork
from .MILmodels import EmbeddingMIL, PyramidalMILmodel, NestedPyramidalMILmodel


class FpnMilInputAdapter(nn.Module):
    """Adapt generic dataloader tensors to FPN-MIL expected input/output format."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def _prepare_inputs(self, images):
        # FPN-MIL expects BNCHW. Some albumentations paths produce BHWC.
        if images.dim() == 4:
            if images.shape[-1] == 3 and images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2).contiguous()
            return images.unsqueeze(1)
        if images.dim() == 5:
            if images.shape[-1] == 3 and images.shape[2] != 3:
                images = images.permute(0, 1, 4, 2, 3).contiguous()
            return images
        raise ValueError(f"Unsupported FPN-MIL input shape: {tuple(images.shape)}")

    def _primary_output(self, output):
        # Generic trainer expects logits tensor. Keep only primary prediction.
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def forward(self, images):
        return self._primary_output(self.base_model(self._prepare_inputs(images)))

    def encode_features(self, images):
        images = self._prepare_inputs(images)
        if self.base_model.inst_encoder is None:
            return images

        if getattr(self.base_model, "multi_scale_model", None) in ["fpn", "backbone_pyramid"]:
            batch_size, num_patches, channels, height, width = images.size()
            flattened_images = images.view(-1, channels, height, width)
            inst_encoder = self.base_model.inst_encoder
            if isinstance(inst_encoder, FeaturePyramidNetwork) and inst_encoder.backbone is not None:
                feature_maps = inst_encoder.backbone(flattened_images)
                feature_maps = list(feature_maps.values()) if isinstance(feature_maps, dict) else list(feature_maps)
                expected_channels = [
                    block[0].in_channels
                    for _, block in sorted(inst_encoder.inner_blocks.items(), key=lambda item: item[0])
                ]
                matched_maps = []
                search_start = 0
                for channels in expected_channels:
                    match_index = next(
                        (idx for idx in range(search_start, len(feature_maps)) if feature_maps[idx].shape[1] == channels),
                        None,
                    )
                    if match_index is None:
                        raise ValueError(
                            f"Could not find FPN feature map with {channels} channels. "
                            f"Available channels: {[fmap.shape[1] for fmap in feature_maps]}"
                        )
                    matched_maps.append(feature_maps[match_index])
                    search_start = match_index + 1
                feature_maps = matched_maps
            else:
                feature_maps = inst_encoder(flattened_images)
                feature_maps = list(feature_maps.values()) if isinstance(feature_maps, dict) else list(feature_maps)

            return [
                fmap.view(batch_size, num_patches, fmap.size(1), fmap.size(2), fmap.size(3))
                for fmap in feature_maps
            ]

        batch_size, num_patches, channels, height, width = images.size()
        features = self.base_model.inst_encoder(images.view(-1, channels, height, width))
        return features.view(batch_size, num_patches, -1)

    def classify_features(self, features):
        inst_encoder = self.base_model.inst_encoder
        if isinstance(inst_encoder, FeaturePyramidNetwork):
            original_backbone = inst_encoder.backbone
            inst_encoder.backbone = None
            try:
                return self._primary_output(self.base_model(features))
            finally:
                inst_encoder.backbone = original_backbone

        self.base_model.inst_encoder = None
        try:
            return self._primary_output(self.base_model(features))
        finally:
            self.base_model.inst_encoder = inst_encoder

    def freeze_image_encoder(self):
        return self.base_model.freeze_image_encoder()


def build_model(args): 

    ###################### Define the Feature Extractor ######################
    if args.feature_extraction in ['online', 'both']: 
        feature_extractor, num_chs = Define_Feature_Extractor(args) 

    else: # offline feature extraction: use pre-extracted features 
            
        if args.multi_scale_model in ['backbone_pyramid', 'fpn']: # FPN-based instance encoder for fpn-based mil models
            
            feature_extractor = FeaturePyramidNetwork(
                backbone=None, 
                scales=args.scales,                            
                out_channels=args.fpn_dim,                           
                top_down_pathway = True if args.multi_scale_model == 'fpn' else False,                                    
                upsample_method = args.upsample_method,      
                norm_layer = args.norm_fpn,
                arch=args.arch,
            )
                
            num_chs = args.fpn_dim 
            
        else: # if single/multi-scale patch-based mil models 
            feature_extractor = None # directly use the pre-extracted features 
            num_chs = args.feat_dim # feat dim of pre-extracted features 
           
    ########################### Define the MIL Model ###########################
    mil_args = dict(training_strategy=args.training_strategy,
                    is_training = args.train, #not args.roi_eval, 
                    multi_scale_model = args.multi_scale_model,  
                    inst_encoder=feature_extractor,
                    embedding_size=num_chs,
                    sigmoid_func = False, 
                    num_classes=args.n_class,
                    drop_classhead=args.drop_classhead,
                    map_prob_func = args.map_prob_func,
                    # MIL Encoder args
                    type_mil_encoder = args.type_mil_encoder,
                    fcl_encoder_dim = args.fcl_encoder_dim, 
                    fcl_dropout = args.fcl_dropout if args.type_mil_encoder == 'mlp' else None, 
                    sab_num_heads = args.sab_num_heads if args.type_mil_encoder == 'sab' else None, 
                    isab_num_heads = args.isab_num_heads if args.type_mil_encoder == 'isab' else None,
                    num_encoder_blocks = args.num_encoder_blocks if args.type_mil_encoder in ['sab', 'isab'] else None, 
                    # MIL Aggregator args
                    pooling_type=args.pooling_type,
                    fcl_attention_dim=args.fcl_attention_dim,
                    drop_attention_pool=args.drop_attention_pool,
                    pma_num_heads = args.pma_num_heads if args.pooling_type == 'pma' else None, 
                    # General self-attention based args 
                    drop_mha=args.drop_mha if args.type_mil_encoder in ['isab', 'sab'] else None, 
                    trans_layer_norm=args.trans_layer_norm if args.type_mil_encoder in ['isab', 'sab'] else None
                   )

    # instantiate MIL Model
    if args.mil_type == 'embedding': # single-scale patch-based mil models 
        model = EmbeddingMIL(mil_type = args.mil_type, 
                             num_inst = [math.ceil(args.img_size[0]/s) * math.ceil(args.img_size[1]/s) for s in args.scales], # number of instances (patches) per image 
                             mil_args = mil_args)
        
    elif args.mil_type == 'pyramidal_mil':

        if args.nested_model: # nested MIL formulation (multi-level aggregation) 
            
            # number of instances per scale for each aggregation level 
            num_inst = [(args.patch_size/s)**2 for s in args.scales]
            num_inst.append(math.ceil(args.img_size[0]/args.patch_size) * math.ceil(args.img_size[1]/args.patch_size))
            
            model = NestedPyramidalMILmodel(
                args.type_scale_aggregator, 
                args.type_region_encoder, 
                args.type_region_pooling, 
                deep_supervision = args.deep_supervision, 
                scales = args.scales, 
                num_inst = num_inst,
                mil_args = mil_args
            )
            
        else: # convetional MIL formulation (globally group and aggregate all instances under the same bag for each scale) 
            # number of instances per scale
            if args.multi_scale_model in ['fpn', 'backbone_pyramid']: # FPN-based mil models 
                num_patches = math.ceil(int(args.img_size[0])/int(args.patch_size)) * math.ceil(int(args.img_size[1])/args.patch_size)
                num_inst = [(args.patch_size/s)**2 * num_patches for s in args.scales] 
                
            else: # multi-scale patch-based mil models 
                num_inst = [math.ceil(args.img_size[0]/s) * math.ceil(args.img_size[1]/s) for s in args.scales]
            
            model = PyramidalMILmodel(
                args.type_scale_aggregator, 
                args.deep_supervision, 
                args.scales, 
                num_inst = num_inst,
                mil_args = mil_args
            )

    return model 
