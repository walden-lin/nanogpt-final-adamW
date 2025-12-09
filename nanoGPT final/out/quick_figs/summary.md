# Optimizer comparison summary

Plotted runs:
- quick_adamw_gqa
- quick_adaml2_gqa
- quick_adamw_coswr_gqa
- quick_adaml2_coswr_gqa

Run hyperparameters (from logs):

| run | optim_mode | lr_schedule | warm_restarts | lr | weight_decay | l2_lambda | l2_target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| quick_adamw_gqa | adamw | constant | False | 0.001 | 0.01 | 0.0 | all |
| quick_adaml2_gqa | adam_l2 | constant | False | 0.001 | 0.1 | 0.01 | weights_only |
| quick_adamw_coswr_gqa | adamw | cosine | True | 0.0009986128001799076 | 0.01 | 0.0 | all |
| quick_adaml2_coswr_gqa | adam_l2 | cosine | True | 0.0009986128001799076 | 0.1 | 0.01 | weights_only |

Figures:
- out/quick_figs/loss_train.png
- out/quick_figs/loss_val.png
- out/quick_figs/param_norm.png
- out/quick_figs/grad_norm.png
- out/quick_figs/lr_schedules.png
