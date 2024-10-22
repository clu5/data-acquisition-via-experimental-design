#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 32 --exp_name batch_size_32 --baselines KNNShapley
#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 16 --exp_name batch_size_16 --baselines KNNShapley
#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 8 --exp_name batch_size_8 --baselines KNNShapley
#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 4 --exp_name batch_size_4 --baselines KNNShapley
#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 2 --exp_name batch_size_2 --baselines KNNShapley
#python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 10000 --batch_size 1 --exp_name batch_size_1 --baselines KNNShapley

python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 32 --exp_name batch_size_32 --baselines KNNShapley --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 16 --exp_name batch_size_16 --baselines KNNShapley --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 8 --exp_name batch_size_8 --baselines KNNShapley --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 4 --exp_name batch_size_4 --baselines KNNShapley --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 2 --exp_name batch_size_2 --baselines KNNShapley --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 10000 --batch_size 1 --exp_name batch_size_1 --baselines KNNShapley --skip_save
