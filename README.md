# Structure Aware Event Extraction
Using Structured information to enhance the event extraction performance
This is a Pytorch implementation of BiLSTM-CRF for Named Entity Recognition, which is described in [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)

## Data
generate the tag vocab and word vocab by running this command:
```
python3 vocab.py "/home/ubuntu/ynliang/bilstm_crf_pytorch/data/train.txt" "/home/ubuntu/ynliang/bilstm_crf_pytorch/vocab/sent_vocab.json" "/home/ubuntu/ynliang/bilstm_crf_pytorch/vocab/tag_vocab.json"
```

## Usage
For training the model, you can use the following command:
```
sh run.sh train
```
For those who are not able to use GPU, use the following command to train:
```
sh run.sh train-without-cuda
```
For testing, you can use the following command:
```
sh run.sh test
```
For getting the inference result only, you can use the following command:
python3 bert_crf.py --mode infid
Also, if you have no GPU, you can use the following command(this procedure won't take long time when using CPU):
```
sh run.sh test-without-cuda
```
There is already a trained model in the [model](./model) folder, so you can execute the testing command directly without training.

If you want to change some hyper-parameters, use the following command to refer to the options.
```
python run.py --help
```

Generate the vocab related information
```
ppython3 vocab.py "/home/ubuntu/ynliang/bilstm_crf_pytorch/StructuredEventExtraction/data/train_mavendata.txt" "/home/ubuntu/ynliang/bilstm_crf_pytorch/StructuredEventExtraction/vocab/sent_vocab.json" "/home/ubuntu/ynliang/bilstm_crf_pytorch/StructuredEventExtraction/vocab/tag_vocab.json"
```

## Result
We use `conlleval.pl` to evaluate the model's performance on test data, and
the experiment result on testing data of the trained model is as follows:
```
processed 172601 tokens with 6192 phrases; found: 5660 phrases; correct: 4820.
accuracy:  97.70%; precision:  85.16%; recall:  77.84%; FB1:  81.34
              LOC: precision:  90.45%; recall:  82.31%; FB1:  86.19  2618
              ORG: precision:  78.18%; recall:  75.66%; FB1:  76.90  1288
              PER: precision:  82.38%; recall:  72.83%; FB1:  77.31  1754
```

## Reference
  1. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991)
  2. [cs224n Assignment 4](http://web.stanford.edu/class/cs224n/index.html#schedule)
  3. https://github.com/Dhanachandra/bert_crf
