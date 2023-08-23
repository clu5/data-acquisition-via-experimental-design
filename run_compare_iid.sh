# python compare_valuations.py --dataset mnist --seed 0 --num_validation 100 --scale_data
# python compare_valuations.py --dataset mnist --seed 1 --num_validation 100 --scale_data
# python compare_valuations.py --dataset mnist --seed 2 --num_validation 100 --scale_data
# python compare_valuations.py --dataset mnist --seed 3 --num_validation 100 --scale_data
# python compare_valuations.py --dataset mnist --seed 4 --num_validation 100 --scale_data

python compare_valuations.py --dataset synthetic --seed 0
python compare_valuations.py --dataset synthetic --seed 1
python compare_valuations.py --dataset synthetic --seed 2
python compare_valuations.py --dataset synthetic --seed 3
python compare_valuations.py --dataset synthetic --seed 4

python compare_valuations.py --dataset diabetes --seed 0
python compare_valuations.py --dataset diabetes --seed 1
python compare_valuations.py --dataset diabetes --seed 2
python compare_valuations.py --dataset diabetes --seed 3
python compare_valuations.py --dataset diabetes --seed 4

python compare_valuations.py --dataset housing --seed 0
python compare_valuations.py --dataset housing --seed 1
python compare_valuations.py --dataset housing --seed 2
python compare_valuations.py --dataset housing --seed 3
python compare_valuations.py --dataset housing --seed 4

python compare_valuations.py --dataset wine --scale_data --seed 0
python compare_valuations.py --dataset wine --scale_data --seed 1
python compare_valuations.py --dataset wine --scale_data --seed 2
python compare_valuations.py --dataset wine --scale_data --seed 3
python compare_valuations.py --dataset wine --scale_data --seed 4

python compare_valuations.py --dataset fires --scale_data --seed 0
python compare_valuations.py --dataset fires --scale_data --seed 1
python compare_valuations.py --dataset fires --scale_data --seed 2
python compare_valuations.py --dataset fires --scale_data --seed 3
python compare_valuations.py --dataset fires --scale_data --seed 4

