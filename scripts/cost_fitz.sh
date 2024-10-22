python main.py -d fitzpatrick --num_buyers 100 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func squared --num_seller 15000 --exp_name noise_3
python main.py -d fitzpatrick --num_buyers 100 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func linear --num_seller 15000 --exp_name noise_3
python main.py -d fitzpatrick --num_buyers 100 --noise_level 0.3 --cost_range 1 2 3 4 5 --cost_func square_root --num_seller 15000 --exp_name noise_3

#python main.py -d fitzpatrick --num_buyers 10 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func squared --num_seller 12000 --baselines KNNShapley --skip_save
#python main.py -d fitzpatrick --num_buyers 10 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func square_root --num_seller 12000 --baselines KNNShapley --skip_save
#python main.py -d fitzpatrick --num_buyers 10 --noise_level 0.5 --cost_range 1 2 3 4 5 --cost_func linear  --num_seller 12000 --baselines KNNShapley --skip_save
