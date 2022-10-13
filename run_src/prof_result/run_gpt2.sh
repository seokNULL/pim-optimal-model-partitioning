#!/bin/bash

Help()
{
   # Display Help
   echo "Run profile."
   echo "Syntax: ./run_profile.sh [-h | -s <sequence_length> | -m <model_name> ]"
   echo "Model name: bert/roberta/gpt2"
   echo "To run seq from 8 to 64, just denote model."
   echo
}
# Help

# PYTHON_FILE=/home/jwlee/transformers/new_pim_profiler.py
PYTHON_FILE=/home/jwlee/transformers/gpt2_pim_profiler.py
PYTHON_OPT_FILE=/home/jwlee/transformers/gpt2_opt_profiler.py
echo "Python file: ${PYTHON_FILE}"
echo "Python file: ${PYTHON_OPT_FILE}"


while getopts ":h:m:s:" option; do
	case $option in
	h) # display Help
		Help
		exit;;
	m)
		m=${OPTARG}
		if 		[ ${m} == "bert" ]; then
			echo "bert"
			model_name='bert-small-sequence-class-inferred-'
		elif 	[ ${m} == "roberta" ]; then
			echo "roberta"
			model_name='roberta-sequence-classification-9-inferred-'
		elif 	[ ${m} == "gpt2" ]; then
			echo "gpt2"
			#model_name='gpt2-lm-head-10-inferred-'
			model_name='gpt2-10-inferred-'
		else
			echo "Not implemented"
		fi
		;;
	s)
		s=${OPTARG}
		;;
	*) 
		echo "Need model for running."
		exit 1
	esac
done

MODEL_DIR=/home/jwlee/transformers/onnx_models/${m}
echo "Model directory: ${MODEL_DIR}"

if [[ "${s}" ]]; then
	FILE_DIR=${m}
	if [ ! -d $FILE_DIR ]; then mkdir $FILE_DIR; fi
	cd $FILE_DIR 
    echo "Running ${m} with sequence length ${s}."
    LOG_DIR="seq"_${s}
    if [ -d $LOG_DIR ]; then rm -rf $LOG_DIR; fi
    mkdir $LOG_DIR
    cd $LOG_DIR
    MODEL=$MODEL_DIR/$model_name${s}".onnx"
    cmd="python3 ${PYTHON_FILE} --model ${MODEL} --batch_size 1 --sequence_length ${s} --samples 1 --thread_num 4 --basic_optimization --kernel_time_only --use_pim &> ${m}_${s}.log"
    eval $cmd
    cmd="python3 ${PYTHON_OPT_FILE} --model ${MODEL} --batch_size 1 --sequence_length ${s} --samples 1 --thread_num 4 --basic_optimization --kernel_time_only --use_pim &> ${m}_${s}_opt.log"
    eval $cmd
    cd -
else
	FILE_DIR=${m}
	if [ ! -d $FILE_DIR ]; then mkdir $FILE_DIR; fi
	cd $FILE_DIR 
    for ((s=8;s<=64;s=s*2))
    do
    	echo "Running ${m} with sequence length ${s}."
    	LOG_DIR="seq"_${s}
    	if [ -d $LOG_DIR ]; then rm -rf $LOG_DIR; fi
    	mkdir $LOG_DIR
    	cd $LOG_DIR
    	MODEL=$MODEL_DIR/$model_name${s}".onnx"
    	cmd="python3 ${PYTHON_FILE} --model ${MODEL} --batch_size 1 --sequence_length ${s} --samples 1 --thread_num 4 --basic_optimization --kernel_time_only --use_pim &> ${m}_${s}.log"
    	eval $cmd
    	cmd="python3 ${PYTHON_OPT_FILE} --model ${MODEL} --batch_size 1 --sequence_length ${s} --samples 1 --thread_num 4 --basic_optimization --kernel_time_only --use_pim &> ${m}_${s}_opt.log"
    	eval $cmd
    	cd -
    done
fi

