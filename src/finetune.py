from datetime import datetime
import json
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run finetuning experiments')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Directory for input data')
    parser.add_argument('--results_dir', default="../finetune_results", help='Directory to store results')
    parser.add_argument('--num_buyer', type=int, default=50, help='Number of buyers')
    parser.add_argument('--num_samples', type=int, default=4000, help='Number of samples')
    parser.add_argument('--Ks', type=int, nargs='+', default=[2, 5, 10, 25, 50, 75, 100], help='List of K values')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--num_iters', type=int, default=500, help='Number of iterations')
    parser.add_argument('--model_name', type=str, default='gpt2', choices=['gpt2', 'bert'], help='Model architecture to finetune')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--grad_steps', type=int, default=1, help='Gradient steps')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to reduce runtime')
    parser.add_argument('--tag', type=str, default='', help='Custom tag to add to the filename')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs for CUDA, e.g., "0,1,2"')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    import utils # import pytorch after setting cuda devices

    # Modify parameters for debug mode
    if args.debug:
        args.num_buyer = 2 # Reduce the number of trials
        args.num_samples = 1000  # Reduce the number of samples
        args.epochs = 5         # Reduce the number of epochs
        args.num_iters = 50     # Reduce the number of iterations
        args.Ks = [2, 5]

    # Run experiments
    results = utils.run_exp(
        data_dir=args.data_dir,
        num_buyer=args.num_buyer,
        num_samples=args.num_samples,
        Ks=args.Ks,
        epochs=args.epochs,
        num_iters=args.num_iters,
        model_name=args.model_name,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_steps=args.grad_steps,
        weight_decay=args.weight_decay,
    )

    # Ensure the results directory exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    # Generate a unique filename with date, time and tag
    date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    filename = f"results_{args.model_name}_{date_time}{tag}.json"
    if args.debug:
        filename = f"results_{args.model_name}_debug.json"
    results_path = os.path.join(args.results_dir, filename)

    # Save results to JSON file
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4, default=float)
