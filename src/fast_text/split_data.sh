#!/usr/local/bin/bash


# split the datafile 'cooking.stackexchange.txt' into training and validation
# datasets
head -n 12404 'cooking.stackexchange.txt' > 'cooking.train'
tail -n 3000 'cooking.stackexchange.txt' > 'cooking.valid'

model.test("cooking.valid")