#!/bin/sh

# Main driver script for the adaptation task.

time python src/adaptation_task.py $1

# in case of errors
echo 'rm -rf src/vectors'
rm -rf src/vectors