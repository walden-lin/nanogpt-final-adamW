# Optimizer comparison summary

Plotted runs:
- adamw_gqa_constant
- adaml2_gqa_constant
- adamw_gqa_coswr
- adaml2_gqa_coswr

Run hyperparameters (from logs):

| run | optim_mode | lr_schedule | warm_restarts | lr | weight_decay | l2_lambda | l2_target |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adamw_gqa_constant | adamw | constant | False | 0.001 | 0.01 | 0.0 | all |
| adaml2_gqa_constant | adam_l2 | constant | False | 0.001 | 0.1 | 0.01 | weights_only |
| adamw_gqa_coswr | adamw | cosine | True | 0.0009986128001799076 | 0.01 | 0.0 | all |
| adaml2_gqa_coswr | adam_l2 | cosine | True | 0.0009986128001799076 | 0.1 | 0.01 | weights_only |

Figures:
- out/figs_noquick/loss_train.png
- out/figs_noquick/loss_val.png
- out/figs_noquick/param_norm.png
- out/figs_noquick/grad_norm.png
- out/figs_noquick/lr_schedules.png
