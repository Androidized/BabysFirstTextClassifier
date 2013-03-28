#!/bin/bash

# a script to run a full submission, including blending
python convert.py
python extract.py --binarize
#python extract.py --tfidf --select-features pct --k 20
#python extract.py --tfidf --select-features k-best --k 3000 --nmf 100
python model.py
python blend.py --ensemble

