# STraTS: Self-Supervised Transformer for Sparse and Irregularly Sampled Multivariate Clinical Time-Series
This repo contains an official re-implementation of STraTS in pytorch. <br>
Paper: https://arxiv.org/pdf/2107.14293.pdf

## Important changes
- We included implementations for the following models: [STraTS](https://arxiv.org/pdf/2107.14293.pdf), [GRU-D](https://arxiv.org/pdf/1606.01865.pdf), [InterpNet](https://openreview.net/pdf?id=r1efr3C9Ym), GRU, [TCN](https://arxiv.org/pdf/1803.01271.pdf), [SaND](https://dl.acm.org/doi/pdf/10.5555/3504035.3504536)
- For STraTS, we removed LayerNorm and replaced ReLU activations in the FFN with GELU, as this improved the performance.
- We used mostly similar hyperparameters for both the datasets, and used the same hidden dimension of 64 for all models.
- Taking inspiration from GRU-D, which uses the same dropout mask at each input time-step, resulting in masking out some variables, we also drop a fraction of variables from the input while training STraTS.
- These changes, along with some possible differences arising from Pytorch's inbuilt modules, make some of the results deviate from the original numbers published in the paper.

## Conda env setup
```
conda create -n strats python=3.10.9
source activate strats
pip install pytz pandas tqdm matplotlib 
pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.35.2
pip install scikit-learn==1.2.2
```

## Dataset preprocessing
Download PhysioNet2012 dataset from https://physionet.org/content/challenge-2012/1.0.0/. <br>
Download MIMIC-III from https://physionet.org/content/mimiciii/1.4/, <br>
Update "RAW_DATA_PATH" variable in the preprocessing scripts and run them.
```
python preprocess_physionet_2012.py
python preprocess_mimic_iii_large.py
```

## Training and evaluation
The shell script run_main.sh contains the commands for training and evaluating each of the supported models.
```
chmod +x run_main.sh
./run_main.sh
```

## Results from run_main.sh
![image](https://github.com/sindhura97/STraTS/assets/42525474/25514af6-47a3-4a0b-8861-235217181abd|width=100)

## Citation
If you found this work useful, please cite our paper:
```
@article{tipirneni2022self,
  title={Self-supervised transformer for sparse and irregularly sampled multivariate clinical time-series},
  author={Tipirneni, Sindhu and Reddy, Chandan K},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  volume={16},
  number={6},
  pages={1--17},
  year={2022},
  publisher={ACM New York, NY}
}
```


