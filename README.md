# Data Acquisition via Experimental Design for Data Markets


**Abstract**
> The acquisition of training data is crucial for machine learning applications. Data markets can increase the supply of data, particularly in data-scarce domains such as healthcare, by incentivizing potential data providers to join the market. A major challenge for a data buyer in such a market is choosing the most valuable data points from a data seller. Unlike prior work in data valuation, which assumes centralized data access, we propose a federated approach to the data acquisition problem that is inspired by linear experimental design. Our proposed data acquisition method achieves lower prediction error without requiring labeled validation data and can be optimized in a fast and federated procedure. The key insight of our work is that a method that directly estimates the benefit of acquiring data for test set prediction is particularly compatible with a decentralized market setting.


To appear in NeurIPS 2024.

For more details, see our [paper](https://arxiv.org/abs/2403.13893).

## Install

`pip install -r requirements.txt`

The datasets can be downloaded here 
* Gaussian data -- see `get_gaussian_data` function in `utils.py`
* [MIMIC](https://physionet.org/content/mimiciii/1.4/)
* [RSNA Bone Age](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)
* [Fitzpatrick17K](https://github.com/mattgroh/fitzpatrick17k)
* [DrugLib](https://archive.ics.uci.edu/dataset/461/drug+review+dataset+druglib+com)
