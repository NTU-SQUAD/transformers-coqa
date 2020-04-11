# transformers-coqa

## train

```
# put coqa-train-v1.0.json and coqa-dev-v1.0.json in /data
python run_coqa.py --model_type albert \
                   --model_name_or_path albert-base-v1 \
                   --do_train \
                   --do_eval \
                   --data_dir data/ \
                   --train_file coqa-train-v1.0.json \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 3 \
                   --output_dir model/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 16  \
                   --max_grad_norm -1 \
                   --weight_decay 0.01
```