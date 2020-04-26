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
3. or run `run_coqa.py` with parmeters,for example add adversarial trainging, evaluate process will be done after training
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

## Parameters

## Model explanation


## References