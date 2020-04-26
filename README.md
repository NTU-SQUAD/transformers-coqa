# transformers-coqa

## Requirements
```
python 3.7
# install packages below or just run `pip install -r requirements.txt` 
pip install pytorch
pip install transformers
pip install spacy
pip install tqdm
pip install multiprocessing
pip install tensorboard
pip install tensorboardX

# install language model for spacy
python -m spacy download en
```

## Folder Structure

## Data
`coqa-dev-v1.0.json` for training, and `coqa-dev-v1.0.json` for evaluate.
You can get newest dataset from [CoQA](https://stanfordnlp.github.io/coqa/) 

## Run-train
1. put coqa-train-v1.0.json and coqa-dev-v1.0.json in same folder, for example in folder `data`
2. if you want to running the code using the default settings, just run
    ```
    . ./run.sh
    ```
3. or run `run_coqa.py` with parmeters,for example add adversarial training, and evaluate process will be done after training
    ```
    python run_coqa.py --model_type albert \
                   --model_name_or_path albert-base-v2 \
                   --do_train \
                   --do_eval \
                   --data_dir data/ \
                   --train_file coqa-train-v1.0.json \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir albert-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 \
                   --adversarial
    ```

## Run-evaluate
After you get the prediction files, you can run evaluate seperately.
The evaluation script is provided by CoQA.
To evaluate, just run
```
python evaluate.py --data-file <path_to_dev-v1.0.json> --pred-file <path_to_predictions>
# if your trained the model using default parameters, it will be
python evaluate.py --data-file data/coqa-dev-v1.0.json --pred-file albert-output/predictions_.json
```

## Results

Some commom parameters:

`adam_epsilon=1e-08, data_dir='data/', do_lower_case=True, doc_stride=128,  fp16=False,  , history_len=2, learning_rate=3e-05, max_answer_length=30, max_grad_norm=-1.0, max_query_length=64, max_seq_length=512,  per_gpu_eval_batch_size=8, seed=42, train_file='coqa-train-v1.0.json', warmup_steps=2000, weight_decay=0.01,num_train_epochs=2`



| Model               | Em   | F1   | Parameters                                                   |
| ------------------- | ---- | ---- | ------------------------------------------------------------ |
| bert-base-uncased   | 68.5 | 78.4 | per_gpu_train_batch_size=16                                  |
| roberta-base        | 72.2 | 81.6 | per_gpu_train_batch_size=16                                  |
| albert-base-v2      | 71.5 | 81.0 | per_gpu_train_batch_size=12                                  |
| albert-base-v2 + AT | 71.7 | 81.3 | per_gpu_train_batch_size=8                                   |
| roberta-large       | 76.3 | 85.7 | per_gpu_train_batch_size=3                                   |
| albert-xxlarge-v1   | 79.1 | 88.1 | per_gpu_train_batch_size=2,gradient_accumulation_steps=12, weight_decay=0 |



## Parameters

## Model explanation


## References