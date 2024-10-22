#python main.py -d gaussian --num_buyers 30 --num_dim 30 --num_seller 1000 --baselines DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator
#python main.py -d mimic --num_buyers 30 --num_seller 1000 --baselines DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator
python main.py -d bone-clip --num_buyers 10 --num_seller 300  --baselines DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator
