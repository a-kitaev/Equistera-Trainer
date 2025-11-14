# Default runtime settings for training

default_scope = 'mmpose'

# hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

# custom hooks
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
]

# multi-processing backend
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# visualizer
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    # Uncomment to use Weights & Biases
    # dict(type='WandbVisBackend', 
    #      init_kwargs=dict(project='horse-pose', name='experiment')),
]

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

# logger
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
    num_digits=6)

log_level = 'INFO'
load_from = None
resume = False

# training/validation/testing configs
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=10)
val_cfg = dict()
test_cfg = dict()

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)
