#!/bin/sh

# Main driver script for the primary task.

time python src/primary_task.py $1

# in case of errors
echo 'rm -rf src/vectors'
rm -rf src/vectors