#!/bin/sh
# background run: nohup bash run_officehome.sh 0 2023 > officehome.log 2>&1 &

# normal run: bash run_officehome.sh 0 2023 
# Make sure that the script receives two arguments: GPU_ID and SEED
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <gpu_id> <seed>"
  exit 1
fi

# Assign arguments to variables
gpu_id=$1
seed=$2

# Set the variable s (in this case, hardcoded to 0)
# original: s<=3
for ((s=0;s<=0;s++))
do

## Run the Python scripts with the provided arguments
python image_source.py --gpu_id $gpu_id --seed $seed --output "ckps/s$seed" --dset office-home --max_epoch 50 --s $s

echo "-------------------------------------------------------------------------------------------------"
echo "Domain Adaptation from Source to Target"
echo "Running NCL"
echo "-------------------------------------------------------------------------------------------------"
python image_NCL.py --gpu_id $gpu_id --seed $seed --da uda --dset office-home --output "ckps/t_NCL$seed" --output_src "ckps/s$seed" --s $s --k 3 --cls_par 0.1 --max_epoch 50 --if_teacher 1

#python image_mixmatch.py --gpu_id $gpu_id --seed $seed --da uda --dset office-home --output "ckps/mm$seed" --s $s --output_tar "ckps/t$seed" --cls_par 0.3 --ssl 0.6 --choice ent --ps 0.0 --max_epoch 50

done
