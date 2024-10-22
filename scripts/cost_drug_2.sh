#python main.py -d drug --num_buyers 100 --num_val 40 --num_seller 4000 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func squared --exp_name noise_3
#python main.py -d drug --num_buyers 100 --num_val 40 --num_seller 4000 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func square_root --exp_name noise_3
#python main.py -d drug --num_buyers 100 --num_val 40 --num_seller 4000 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func linear --exp_name noise_3

#python main.py -d drug --num_buyers 100 --num_seller 3500 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func squared --exp_name noise_3
python main.py -d drug --num_buyers 100 --num_seller 3500 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func square_root --exp_name noise_3
python main.py -d drug --num_buyers 100 --num_seller 3500 --noise_level 0.3 --cost_range 1 2 3 4 5 --baselines DVRL DataOob InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --cost_func linear --exp_name noise_3
