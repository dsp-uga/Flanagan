#!/bin/bash
IFS=$'\n'
for l in $(cat $1); do
    gsutil -m cp gs://uga-dsp/project4/data/$l.tar .
    tar xvC ../Data/Test/ -f $l.tar
done
exit 0
