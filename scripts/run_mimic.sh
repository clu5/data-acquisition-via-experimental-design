python main.py -d mimic --num_buyers 100 --num_seller 35000 --baselines DataOob DVRL InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --exp_name main_table
python main.py -d mimic --num_buyers 100 --num_seller 1000 --baselines DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator --exp_name main_table
