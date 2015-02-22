#!/bin/bash

for i in {1..20}
do
	index=$RANDOM
	index=$[ $index % 2]
	index=$((index + 1)) 
	if (( $index == 1 ))
	then
		echo "low beep"
		afplay ../../data/data_preprocessing/300Hz.wav
	else
		echo "high beep"
		afplay ../../data/data_preprocessing/900Hz.wav
	fi		
	sleep 2.3
done
