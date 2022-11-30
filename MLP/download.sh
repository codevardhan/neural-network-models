#!/usr/bin/env bash
# downloading mnist data from deep-ai
mkdir data
wget https://data.deepai.org/mnist.zip
mv mnist.zip ./data
unzip data/mnist.zip -d ./data/ 
rm data/mnist.zip
