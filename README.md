# transformers-coqa

## Requirements

```bash
python 3.7
# install packages below or just run `pip install -r requirements.txt` 
pip install transformers
pip install spacy
pip install tqdm
pip install tensorboard
pip install tensorboardX

# install language model for spacy
python -m spacy download en
```



## Data

`coqa-dev-v1.0.json` for training, and `coqa-dev-v1.0.json` for evaluate.
You can get newest dataset from [CoQA](https://stanfordnlp.github.io/coqa/) 

## Run-train

1. put coqa-train-v1.0.json and coqa-dev-v1.0.json in same folder, for example in folder `data`
2. if you want to running the code using the default settings, just run`./run.sh`
3. or run `run_coqa.py` with parmeters,for example add adversarial training, and evaluate process will be done after training

    ```bash
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
                   --adversarial \
                   --threads 8
    ```

4. To reproduce the result, we also provide shell scripts for all models:

   ```
   # bert
   . ./run_bert.sh
   # roberta, roberta-large
   . ./run_roberta.sh
   . ./run_roberta_large.sh
   # albert, albert-xxlarge, albert-at
   . ./run.sh
   . ./run_albert_xxlarge
   . ./run_albert_at
   ```

   

## Run-evaluate

After you get the prediction files, you can run evaluate seperately.
The evaluation script is provided by CoQA.
To evaluate, just run

```bash
python evaluate.py --data-file <path_to_dev-v1.0.json> --pred-file <path_to_predictions>
# if your trained the model using default parameters, it will be
python evaluate.py --data-file data/coqa-dev-v1.0.json --pred-file albert-output/predictions_.json
```

## Results

Some commom parameters:
`adam_epsilon=1e-08, data_dir='data/', do_lower_case=True, doc_stride=128,  fp16=False, history_len=2, learning_rate=3e-05, max_answer_length=30, max_grad_norm=-1.0, max_query_length=64, max_seq_length=512,  per_gpu_eval_batch_size=8, seed=42, train_file='coqa-train-v1.0.json', warmup_steps=2000, weight_decay=0.01,num_train_epochs=2`

| Model               | Em   | F1   | Parameters                                                   |
| ------------------- | ---- | ---- | ------------------------------------------------------------ |
| bert-base-uncased   | 68.5 | 78.4 | per_gpu_train_batch_size=16                                  |
| roberta-base        | 72.2 | 81.6 | per_gpu_train_batch_size=16                                  |
| albert-base-v2      | 71.5 | 81.0 | per_gpu_train_batch_size=8                                   |
| albert-base-v2 + AT | 71.7 | 81.3 | per_gpu_train_batch_size=8                                   |
| roberta-large       | 76.3 | 85.7 | per_gpu_train_batch_size=3                                   |
| albert-xxlarge-v1   | 79.1 | 88.1 | per_gpu_train_batch_size=2, gradient_accumulation_steps=12, weight_decay=0 |



## Parameters

Here we will explain some important parameters, for all trainning parameters, you can find in `run_coqa.py`



| Param name         | Default value | Details                                    |
| ------------------ | ------------- | ------------------------------------------ |
| model_type         | None          | Type of models,such as bert,albert,roberta |
| model_name_or_path | None          |                                            |
|                    |               |                                            |
|                    |               |                                            |
|                    |               |                                            |
|                    |               |                                            |
|                    |               |                                            |
|                    |               |                                            |
|                    |               |                                            |



## Model explanation

The following is the overview of the whole repo structure, we keep the structure similiar with the `transformers` fine-tune on `SQuAD`.

```bash
├── data
│   ├── coqa-dev-v1.0.json  # CoQA Validation dataset
│   ├── coqa-train-v1.0.json    # CoQA training dataset
│   ├── metrics
│   │   └── coqa_metrics.py # compute the predictions for evaluation
│   └── processors
│       ├── coqa.py # Data processing: create examples from the raw dataset, convert examples into features
│       └── utils.py    # data converters for sequence classification data sets.
├── evaluate.py # script used to run the evaluation only, please refer to the above Run-evaluate section
├── LICENSE
├── model 
│   ├── Layers.py # Multiple LinearLayer class used in the downstream QA tasks
│   ├── modeling_albert.py # core ALBERT model class, including all the architecture for the downstream QA tasks
│   ├── modeling_auto.py # generic class that help instantiate one of the question answering model classes, As the bert like model has similiar input and output. Use this can make clean code and fast develop and test. Refer to the same class in transformers library
│   ├── modeling_bert.py # core BERT model class, including all the architecture for the downstream QA tasks
│   └── modeling_roberta.py  # core BERT model class, including all the architecture for the downstream QA tasks
├── README.md # This instruction you are reading now
├── requirements.txt # The requirements for reproducing our results
├── run_coqa.py # Main function script
├── run.sh # run training with default setting
└── utils
    ├── adversarial.py # class for adversarial Projected gradient descent and fast graident method
    └── tools.py # function used to calculate model parameter numbers
```

## References

1. [coqa-baselines](https://github.com/stanfordnlp/coqa-baselines)
2. [transformers](https://github.com/huggingface/transformers)
3. [bert4coqa](https://github.com/adamluo1995/Bert4CoQA)
4. [SDNet](https://github.com/microsoft/SDNet)

5. [Adversarial Training](https://fyubang.com/2019/10/15/adversarial-train/)