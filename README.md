<div align="center">

# Variational Autoencoder with Label Smoothing for Collaborative Filtering

**[Woo-Seong Yun](https://scholar.google.com/citations?user=ZRXyvtMAAAAJ)** &nbsp;·&nbsp; **Yeong-Hyeon Kim** &nbsp;·&nbsp; **Chan-Woo Jeong** &nbsp;·&nbsp; **Yeo-Jun Choi** &nbsp;·&nbsp; **Yoon-Sik Cho**

<sub>Department of Artificial Intelligence, Chung-Ang University</sub>

*KIIT Fall Conference (The Korean Institute of Information Technology), pp. 162–165, Nov. 2024*

[![Paper](https://img.shields.io/badge/Paper-DBpia-1E4DB7)](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12024885)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

</div>

This is the PyTorch implementation for our VAE-LS paper:

> Variational Autoencoder with Label Smoothing for Collaborative Filtering (KIIT Fall Conference, 2024)

## Overview

Mult-VAE optimizes a multinomial likelihood with a softmax over items to reflect item rankings, but the softmax tends to assign excessive probability to a few specific items, which damages generalization. Inspired by label smoothing in image classification, **VAE-LS** replaces the one-hot interaction target $x_u$ in the Mult-VAE objective with a smoothed target controlled by a strength $\alpha$, which discourages over-confident predictions on specific items and encourages the model to explore a more diverse set of items. On MovieLens20M and Netflix, VAE-LS outperforms linear and non-linear baselines on most metrics, and a sensitivity study shows that light smoothing ($\alpha = 0.05$) gives the best balance between recommendation accuracy and item diversity.

## Requirements

```bash
pip install torch numpy pandas scipy tensorboardX easydict pytz
```

## Datasets

We follow the preprocessing and the standard train/validation/test split of [Mult-VAE (Liang et al., 2018)](https://github.com/dawenl/vae_cf). Ratings ≥ 4 are treated as implicit feedback and users with fewer than 5 interactions are removed.

| Dataset | Users | Items | Interactions | Held-out users (val / test) | Download |
| --- | ---: | ---: | ---: | ---: | --- |
| MovieLens20M | 136,677 | 20,108 | ~10.0M | 10,000 / 10,000 | [ml-20m.zip](http://files.grouplens.org/datasets/movielens/ml-20m.zip) |
| Netflix Prize | 443,435 | 17,769 | ~56.9M | 40,000 / 40,000 | [Kaggle](https://www.kaggle.com/netflix-inc/netflix-prize-data) |

Download each dataset and run the preprocessing notebook in `datasets/`.

## Training

The model is implemented as a single notebook, `MultVAE-LS.ipynb`. Open it and run all cells.

## Results

Bold marks the best result for each metric (Table 1 of the paper).

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">Model</th>
      <th colspan="3" align="center">ML20M</th>
      <th colspan="3" align="center">Netflix</th>
    </tr>
    <tr>
      <th align="center">N@100</th>
      <th align="center">R@50</th>
      <th align="center">R@20</th>
      <th align="center">N@100</th>
      <th align="center">R@50</th>
      <th align="center">R@20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">WMF</td>
      <td align="center">0.386</td>
      <td align="center">0.498</td>
      <td align="center">0.360</td>
      <td align="center">0.351</td>
      <td align="center">0.404</td>
      <td align="center">0.316</td>
    </tr>
    <tr>
      <td align="left">SLIM</td>
      <td align="center">0.401</td>
      <td align="center">0.495</td>
      <td align="center">0.370</td>
      <td align="center">0.379</td>
      <td align="center">0.428</td>
      <td align="center">0.347</td>
    </tr>
    <tr>
      <td align="left">CDAE</td>
      <td align="center">0.418</td>
      <td align="center">0.523</td>
      <td align="center">0.391</td>
      <td align="center">0.376</td>
      <td align="center">0.428</td>
      <td align="center">0.343</td>
    </tr>
    <tr>
      <td align="left">Mult-VAE</td>
      <td align="center">0.426</td>
      <td align="center"><b>0.536</b></td>
      <td align="center">0.395</td>
      <td align="center">0.386</td>
      <td align="center">0.440</td>
      <td align="center">0.348</td>
    </tr>
    <tr>
      <td align="left"><b>VAE-LS</b></td>
      <td align="center"><b>0.430</b></td>
      <td align="center">0.535</td>
      <td align="center"><b>0.399</b></td>
      <td align="center"><b>0.388</b></td>
      <td align="center"><b>0.443</b></td>
      <td align="center"><b>0.355</b></td>
    </tr>
  </tbody>
</table>

Sensitivity to the smoothing strength $\alpha$ (Table 2 of the paper):

<table>
  <thead>
    <tr>
      <th rowspan="2" align="left">α</th>
      <th colspan="3" align="center">ML20M</th>
      <th colspan="3" align="center">Netflix</th>
    </tr>
    <tr>
      <th align="center">N@100</th>
      <th align="center">R@50</th>
      <th align="center">R@20</th>
      <th align="center">N@100</th>
      <th align="center">R@50</th>
      <th align="center">R@20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="left">0.01</td>
      <td align="center">0.429</td>
      <td align="center">0.533</td>
      <td align="center">0.397</td>
      <td align="center"><b>0.388</b></td>
      <td align="center"><b>0.444</b></td>
      <td align="center">0.354</td>
    </tr>
    <tr>
      <td align="left">0.05</td>
      <td align="center"><b>0.430</b></td>
      <td align="center"><b>0.535</b></td>
      <td align="center"><b>0.399</b></td>
      <td align="center"><b>0.388</b></td>
      <td align="center">0.443</td>
      <td align="center"><b>0.355</b></td>
    </tr>
    <tr>
      <td align="left">0.1</td>
      <td align="center">0.429</td>
      <td align="center">0.530</td>
      <td align="center"><b>0.399</b></td>
      <td align="center"><b>0.388</b></td>
      <td align="center">0.441</td>
      <td align="center"><b>0.355</b></td>
    </tr>
    <tr>
      <td align="left">0.2</td>
      <td align="center">0.426</td>
      <td align="center">0.528</td>
      <td align="center">0.397</td>
      <td align="center">0.386</td>
      <td align="center">0.438</td>
      <td align="center">0.354</td>
    </tr>
  </tbody>
</table>

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{yun2024vaels,
  title     = {Variational Autoencoder with Label Smoothing for Collaborative Filtering},
  author    = {Woo-Seong Yun and Yeong-Hyeon Kim and Chan-Woo Jeong and Yeo-Jun Choi and Yoon-Sik Cho},
  booktitle = {Proceedings of the KIIT Fall Conference},
  pages     = {162--165},
  year      = {2024},
  month     = nov,
  publisher = {The Korean Institute of Information Technology}
}
```

## Acknowledgements

This research was supported by the MSIT (Ministry of Science and ICT), Korea, under the ITRC (Information Technology Research Center) support program (IITP-2024-RS-2024-00438056) supervised by the IITP (Institute for Information & Communications Technology Planning & Evaluation), and by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00419201).

Our implementation builds on [Mult-VAE](https://github.com/dawenl/vae_cf).
