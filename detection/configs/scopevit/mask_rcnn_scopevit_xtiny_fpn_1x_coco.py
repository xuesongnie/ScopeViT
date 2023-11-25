_base_ = [
    '../_base_/models/mask_rcnn_scopevit_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

# optimizer
model = dict(
    backbone=dict(
        type='ScopeViT_feat',
        arch='xtiny',
        drop_path_rate=0.05,
        init_cfg=dict(
            type='Pretrained', 
            checkpoint='',
            ),
        ),
    neck=dict(
        type='FPN',
        in_channels=[32, 64, 128, 256],
        out_channels=256,
        num_outs=5))
# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0002, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'layer_scale': dict(decay_mult=0.),
                                                 'scale': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=None)

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
evaluation = dict(save_best='auto')
