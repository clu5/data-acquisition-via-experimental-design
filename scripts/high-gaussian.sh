 python main_gaussian.py --num_seller 5000 --num_dim 1000 --num_val 100 --num_iters 100 --alpha 0.1 --baselines DataBanzhaf DataOob KNNShapley LavaEvaluator RandomEvaluator --cluster
 python main_gaussian.py --num_seller 5000 --num_dim 5000 --num_val 100 --num_iters 100 --alpha 0.1 --baselines DataBanzhaf DataOob KNNShapley LavaEvaluator RandomEvaluator --cluster

python main_gaussian.py --num_seller 1000 --num_dim 1000 --num_val 100 --num_iters 100 --alpha 0.1 --baselines AME BetaShapley DataBanzhaf DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator --cluster
python main_gaussian.py --num_seller 1000 --num_dim 1000 --num_val 100 --num_iters 100 --alpha 0.1 --baselines AME BetaShapley DataBanzhaf DataOob DataShapley DVRL InfluenceSubsample KNNShapley LavaEvaluator LeaveOneOut RandomEvaluator
