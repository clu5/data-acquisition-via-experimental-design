python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_0 --reg_lambda 0.0 --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_2 --reg_lambda 0.2 --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_4 --reg_lambda 0.4 --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_6 --reg_lambda 0.6 --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_8 --reg_lambda 0.8 --skip_save
python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_1 --reg_lambda 1.0 --skip_save

python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_0 --reg_lambda 0.0 --skip_save
python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_2 --reg_lambda 0.2 --skip_save
python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_4 --reg_lambda 0.4 --skip_save
python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_6 --reg_lambda 0.6 --skip_save
python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_8 --reg_lambda 0.8 --skip_save
python main.py -d drug --num_buyers 100 --num_seller 3000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_1 --reg_lambda 1.0 --skip_save

python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_0 --reg_lambda 0.0 --skip_save
python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_2 --reg_lambda 0.2 --skip_save
python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_4 --reg_lambda 0.4 --skip_save
python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_6 --reg_lambda 0.6 --skip_save
python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_8 --reg_lambda 0.8 --skip_save
python main.py -d fitzpatrick --num_buyers 100 --num_seller 15000 --baselines KNNShapley RandomEvaluator --exp_name cov_reg_1 --reg_lambda 1.0 --skip_save
