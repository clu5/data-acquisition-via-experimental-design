#python main.py -d bone --num_buyers 100 --num_seller 12000 --baselines DataOob DVRL InfluenceSubsample KNNShapley LavaEvaluator RandomEvaluator --exp_name main_table
python main.py -d bone --num_buyers 10 --num_seller 12000 --baselines DVRL KNNShapley LavaEvaluator RandomEvaluator --exp_name visualize
