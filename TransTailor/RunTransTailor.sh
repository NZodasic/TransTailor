#!/bin/sh

TransTailor_Path=$1

python $TransTailor_Path/TransTailorV2.py --root $1 --numworker 10 --batchsize 128