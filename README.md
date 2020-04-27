# transformers-coqa

## Requirements

```bash
Our code is tested under python 3.7 and pytorch 1.4.1(torch== 1.4.1)
# install packages below or just run `pip install -r requirements.txt` 
pip install transformers==2.8.0
pip install numpy==1.16.4
pip install spacy==2.2.4
pip install tqdm==4.42.1
pip install tensorboard==1.14.0
pip install tensorboardX==2.0

# install language model for spacy
python -m spacy download en
```

## Data

`coqa-train-v1.0.json` for training, and `coqa-dev-v1.0.json` for evaluate.
You can get newest dataset from [CoQA](https://stanfordnlp.github.io/coqa/) **OR**
Direct download [coqa-train-v1.0.json](https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json) and [coqa-dev-v1.0.json](https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json) seperately

## Run-train

1. put coqa-train-v1.0.json and coqa-dev-v1.0.json in same folder, for example in folder `data/`
2. if you want to running the code using the default settings, just run`./run.sh`
3. or run `run_coqa.py` with parmeters,for example add adversarial training, and evaluate process will be done after training

    ```bash
    python3 run_coqa.py --model_type albert \
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

   ```bash
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

5. The estimate training and evaluation time for `albert-base` model on the CoQA task is around **6** hours on the GPU server provided to MSAI students. For the `albert_xxlarge`, `bert`, `roberta` and `roberta_large`, it is expected more time to training

## Run-evaluate

After you get the prediction files, you can run evaluate seperately.
The evaluation script is provided by CoQA.
To evaluate, just run

```bash
python evaluate.py --data-file <path_to_dev-v1.0.json> --pred-file <path_to_predictions>
# if your trained the model using default parameters, it will be
python evaluate.py --data-file data/coqa-dev-v1.0.json --pred-file albert-output/predictions_.json
```

## Docker
We also provide `Dockerfile`, if you have problem with the environment, you can try to build a docker iamge locally and run the code inside docker.

Notice: This is a GPU based image.
```
## build image
docker build -t transformers-coqa .

## run the iamge
docker run -it coqa

## run the code
cd transformer-coqa && \
. run.sh
```

## Results

Some commom parameters:
`adam_epsilon=1e-08, data_dir='data/', do_lower_case=True, doc_stride=128,  fp16=False, history_len=2, learning_rate=3e-05, max_answer_length=30, max_grad_norm=-1.0, max_query_length=64, max_seq_length=512,  per_gpu_eval_batch_size=8, seed=42, train_file='coqa-train-v1.0.json', warmup_steps=2000, weight_decay=0.01,num_train_epochs=2`

Our result:
| Model               | Em   | F1   | Parameters                                                   |
| ------------------- | ---- | ---- | ------------------------------------------------------------ |
| bert-base-uncased   | 68.5 | 78.4 | per_gpu_train_batch_size=16                                  |
| roberta-base        | 72.2 | 81.6 | per_gpu_train_batch_size=16                                  |
| albert-base-v2      | 71.5 | 81.0 | per_gpu_train_batch_size=8                                   |
| albert-base-v2 + AT | 71.7 | 81.3 | per_gpu_train_batch_size=8                                   |
| roberta-large       | 76.3 | 85.7 | per_gpu_train_batch_size=3                                   |
| albert-xxlarge-v1   | 79.1 | 88.1 | per_gpu_train_batch_size=2, gradient_accumulation_steps=12, weight_decay=0 |

The current CoQA [leadboard](https://stanfordnlp.github.io/coqa/)

## Parameters

Here we will explain some important parameters, for all trainning parameters, you can find in `run_coqa.py`

| Param name                  | Default value        | Details                                                      |
| --------------------------- | -------------------- | ------------------------------------------------------------ |
| model_type                  | None                 | Type of models,such as bert,albert,roberta.                  |
| model_name_or_path          | None                 | Path to pre-trained model or model name listed above.        |
| output_dir                  | None                 | The output directory where the model checkpoints and predictions will be written. |
| data_dir                    | None                 | The directory where training and evaluate data (json files) are placed, if  is None, the root directory will be taken. |
| train_file                  | coqa-train-v1.0.json | The input training file.                                     |
| predict_file                | coqa-dev-v1.0.json   | The input evaluation file.                                   |
| max_seq_length              | 512                  | The maximum total input sequence length after WordPiece tokenization. |
| doc_stride                  | 128                  | When splitting up a long document into chunks, how much stride to take between chunks. |
| max_query_length            | 64                   | The maximum number of tokens for the question. Questions longer than this will be truncated to this length. |
| do_train                    | False                | Whether to run training.                                     |
| do_eval                     | False                | Whether to run eval on the dev set.                          |
| evaluate_during_training    | False                | Run evaluation during training at 10times each logging step  |
| do_lower_case               | False                | Set this flag if you are using an uncased model.             |
| per_gpu_train_batch_size    | 8                    | Batch size per GPU/CPU for training.                         |
| learning_rate               | 3e-5                 | The initial learning rate for Adam.                          |
| gradient_accumulation_steps | 1                    | Number of updates steps to accumulate before performing a backward/update pass. |
| weight_decay                | 0.01                 | Weight decay if we apply some.                               |
| num_train_epochs            | 2                    | Total number of training epochs to perform.                  |
| warmup_steps                | 2000                 | Linear warmup over warmup_steps.This should not be too small(such as 200), will may lead to low score in this model. |
| history_len                 | 2                    | keep len of history quesiton-answers                         |
| logging_steps               | 50                   | Log every X updates steps.                                   |
| threads                     | 1                    | multiple threads for converting example to features          |

## Model explanation

The following is the overview of the whole repo structure, we keep the structure similiar with the `transformers` fine-tune on `SQuAD`, we use the `transformers` library to load pre-trained model and model implementation.

```bash
├── data
│   ├── coqa-dev-v1.0.json  # CoQA Validation dataset
│   ├── coqa-train-v1.0.json # CoQA training dataset
│   ├── metrics
│   │   └── coqa_metrics.py # Compute and save the predictions, do evaluation and get the final score
│   └── processors
│       ├── coqa.py # Data processing: create examples from the raw dataset, convert examples into features
│       └── utils.py # data Processor for sequence classification data sets.
├── evaluate.py # script used to run the evaluation only, please refer to the above Run-evaluate section
├── LICENSE
├── model
│   ├── Layers.py # Multiple LinearLayer class used in the downstream QA tasks
│   ├── modeling_albert.py # core ALBERT model class, add architecture for the downstream QA tasks on the top of pre-trained ALBERT model from transformer library.
│   ├── modeling_auto.py # generic class that help instantiate one of the question answering model classes, As the bert like model has similiar input and output. Use this can make clean code and fast develop and test. Refer to the same class in transformers library
│   ├── modeling_bert.py # core BERT model class, including all the architecture for the downstream QA tasks
│   └── modeling_roberta.py # core Roberta model class, including all the architecture for the downstream QA tasks
├── README.md # This instruction you are reading now
├── requirements.txt # The requirements for reproducing our results
├── run_coqa.py # Main function script
├── run.sh # run training with default setting
└── utils
    ├── adversarial.py # class for adversarial Projected gradient descent and fast graident method
    └── tools.py # function used to calculate model parameter numbers
```

The following are detailed descrpition on some core scripts:

- [run_coqa.py](run_coqa.py): This script is the main function script used for training and evaluation. It:
   1. Defines All system parameters and some training parameter
   2. Setup CUDA, GPU, distributed training and logging, all seeds
   3. Instantiate and initialize the corresponding model config, tokenizer and pre-train model
   4. Calculate the number of trainable parameters
   5. Define and execute the training and evaluation function
- [coqa.py](data/processors/coqa.py): This script contains the functions and classes used to conduct data preprocess, it:
   1. Define the data structure of `CoqaExamples`, `CoqaFeatures` and `CoqaResult`
   2. Define the class of `CoqaProcessor`, which is used to process the raw data to get examples. It implements the methods `get_raw_context_offsets` to add word offset, `find_span_with_gt` to find the best answer span, `_create_examples` to convert single conversation (context and QAs pairs) into `CoqaExample`, `get_examples` to parallel execute the create_examples
   3. Define the methods `coqa_convert_example_to_features` to convert `CoqaExamples` into `CoqaFeatures`, `coqa_convert_examples_to_features` to parallel execute `coqa_convert_example_to_features`
- [modeling_albert.py](model/modeling_albert.py): This script contains the core ALBERT class and related downstream CoQA architecture, it:
   1. Load the pre-trained ALBERT model from `transformer` library
   2. Build downstream CoQA tasks architecture on the top of ALBERT last hidden state and pooler output to get the training loss for training and start, end, yes, no, unknown logits for prediction. This architecture is the same as shown in the presentation and report

## References

1. [coqa-baselines](https://github.com/stanfordnlp/coqa-baselines)
2. [transformers](https://github.com/huggingface/transformers)
3. [bert4coqa](https://github.com/adamluo1995/Bert4CoQA)
4. [SDNet](https://github.com/microsoft/SDNet)
5. [Adversarial Training](https://fyubang.com/2019/10/15/adversarial-train/)
