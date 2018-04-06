#!/bin/bash

for name in $(ls $1);do
    python png_extractor.sh -i $2 -n $name
done
