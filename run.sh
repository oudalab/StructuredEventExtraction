#!/bin/sh

if [ "$1" = "train" ]
then
  python run.py train ./data/train_15_982_sampled.txt ./data/dev_15_982_sampled.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json --cuda
elif [ "$1" = "train-without-cuda" ]
then
  python run.py train ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json max-epoch=30
elif [ "$1" = "test" ]
then
  python run.py test ./data/train_15_982_sampled.txt ./result_inf.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth --cuda
elif [ "$1" = "inf" ]
then
  python run.py inf ./data/dev_inf_mavendata.txt ./result_inf.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth --cuda
elif [ "$1" = "infid" ]
then
  python run.py infid ./data/test_mavendata_documentid.txt ./result_inf_docid.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth --cuda
elif [ "$1" = "test-without-cuda" ]
then
  python run.py test ./data/test.txt ./result.txt  ./vocab/sent_vocab.json ./vocab/tag_vocab.json ./model/model.pth
  perl conlleval.pl < result.txt
elif [ "$1" = "vocab" ]
then
	python vocab.py ./data/train.txt ./vocab/sent_vocab.json ./vocab/tag_vocab.json
fi

