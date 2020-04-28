python3 run_coqa.py --model_type albert-xxlarge \
                   --model_name_or_path albert-xxlarge-v1 \
                   --do_train \
                   --do_eval \
                   --data_dir data/ \
                   --train_file coqa-train-v1.0.json \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir albert-xxlarge-output/ \
                   --per_gpu_train_batch_size 2  \
                   --max_grad_norm -1 \
                   --gradient_accumulation_steps 12 \
                   --weight_decay 0
