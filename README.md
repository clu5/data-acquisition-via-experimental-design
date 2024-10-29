# Data Acquisition via Experimental Design for Data Markets


**Abstract**
> The acquisition of training data is crucial for machine learning applications. Data markets can increase the supply of data, particularly in data-scarce domains such as healthcare, by incentivizing potential data providers to join the market. A major challenge for a data buyer in such a market is choosing the most valuable data points from a data seller. Unlike prior work in data valuation, which assumes centralized data access, we propose a federated approach to the data acquisition problem that is inspired by linear experimental design. Our proposed data acquisition method achieves lower prediction error without requiring labeled validation data and can be optimized in a fast and federated procedure. The key insight of our work is that a method that directly estimates the benefit of acquiring data for test set prediction is particularly compatible with a decentralized market setting.


To appear in NeurIPS 2024.

For more details, see our [paper](https://arxiv.org/abs/2403.13893).

## Installation

1. Create and activate Python virtual environment `python -m venv DAVED && source DAVED/bin/activate`
2. Install Python packages `pip install -r requirements.txt`

## Datasets

The datasets can be downloaded here 
* Gaussian data -- see `get_gaussian_data` function in `utils.py`
* [MIMIC](https://physionet.org/content/mimiciii/1.4/)
* [RSNA Bone Age](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
* [Fitzpatrick17K](https://github.com/mattgroh/fitzpatrick17k)
* [DrugLib](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)

After downloading the datasets, place them in a data/ directory with the following structure:
```
data/
├── bone-age/
│   ├── boneage-training-dataset/
│   └── train.csv
├── druglib/
│   └── druglib.csv
├── fitzpatrick17k/
│   ├── images/
│   └── fitzpatrick-mod.csv
└── mimic-los-data.csv
```

## Usage
### Basic Example
```python
import utils
import frank_wolfe

# Generate synthetic data
data = utils.get_gaussian_data(num_samples=1000, dim=10)

# Split into train/test
split_data = utils.split_data(
    num_buyer=1,  # Number of test points
    num_val=10,   # Size of validation set
    X=data['X'],
    y=data['y']
)

# Run data acquisition
results = frank_wolfe.design_selection(
    split_data['X_sell'],  # Seller data
    split_data['y_sell'],
    split_data['X_buy'],   # Buyer test point
    split_data['y_buy'],
    num_select=10,         # Number of points to select
    num_iters=500         # Optimization iterations
)
```

### Running Experiments
```bash
# Basic experiment on Gaussian data
python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 1000

# Experiment with cost constraints
python main.py -d gaussian --num_buyers 100 --num_dim 30 --num_seller 1000 --cost_range 1 2 3 4 5 --cost_func squared

# Fine-tuning experiments
python finetune.py --num_buyer 50 --epochs 25 --model_name gpt2
```

See `main.py` and `finetune.py` for complete list of command line arguments.

**File Structure**
* `main.py`: Main script for running experiments
* `utils.py`: Utility functions for data loading and processing
* `frank_wolfe.py`: Implementation of optimization algorithm
* `convex.py`: Convex optimization solver
* `finetune.py`: Script for fine-tuning experiments

## Citation
If you use this code in your research, please cite our paper:
```
@misc{lu2024daveddataacquisitionexperimental,
      title={DAVED: Data Acquisition via Experimental Design for Data Markets}, 
      author={Charles Lu and Baihe Huang and Sai Praneeth Karimireddy and Praneeth Vepakomma and Michael Jordan and Ramesh Raskar},
      year={2024},
      eprint={2403.13893},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.13893}, 
}
```
