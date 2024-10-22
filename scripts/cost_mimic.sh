python main.py -d mimic --num_buyers 100 --num_seller 35000 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func squared --exp_name noise_3
python main.py -d mimic --num_buyers 100 --num_seller 35000 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func linear --exp_name noise_3
python main.py -d mimic --num_buyers 100 --num_seller 35000 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func square_root --exp_name noise_3

#python main.py -d mimic --num_buyers 10 --num_seller 35000 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func squared --skip_save
#python main.py -d mimic --num_buyers 10 --num_seller 35000 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func square_root --skip_save
#python main.py -d mimic --num_buyers 10 --num_seller 35000 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func linear --skip_save
