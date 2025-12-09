# Experiment Summary Template (Optimizer × Scheduler × GQA)

## Setup
- Model: block_size=___, n_layer=___, n_head=___, n_headgroup=___, n_embd=___, dropout=___
- Training: batch_size=___, max_iters=___, dataset=___, device=___
- Attention: GQA enabled? ___
- Optimizer: {adamw | adam_l2}, lr=___, schedule={constant|step|cosine}, warm_restarts=___, weight_decay=___, l2_lambda=___

## Final Losses
| Run | Train loss | Val loss | Best val loss | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

Best configuration: ___

## Curves
- Train vs val loss: `figures/loss_train.png`, `figures/loss_val.png`
- Parameter norms: `figures/param_norm.png`
- Gradient norms: `figures/grad_norm.png`
- LR schedules: `figures/lr_schedules.png`

## Generated Samples
- AdamW sample: `out/.../sample.txt`
- Adam+L2 sample: `out/.../sample.txt`

## Results and Discussion (bullet notes)
- AdamW vs Adam+L2 losses:
- Parameter norm stability:
- Effect of cosine schedule:
- Effect of warm restarts:

## Next Steps
- Longer runs? ___
- Tune T0/T_mult? ___
- Try different group sizes for GQA? ___
