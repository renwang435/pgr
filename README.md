# Prioritized Generative Replay

[Renhao Wang](https://renwang435.github.io/), [Kevin Frans](https://kvfrans.com/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/), [Alexei A. Efros](http://people.eecs.berkeley.edu/~efros/)

[[`arXiv`](https://arxiv.org/abs/2410.18082)] [[`BibTeX`](#Citing)]

## Installation

This repository is based heavily off the release code of [SynthER](https://github.com/conglu1997/SynthER?tab=readme-ov-file#setup). The instructions for setting up their environment are reproduced below: 

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

Our code is tested on Python 3.8.
If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

## Running Instructions

```bash
python synther/online/online_cond.py --env quadruped-walk-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.1'
```

```bash
python synther/online/online_cond.py --env cheetah-run-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25'
```

```bash
python synther/online/online_cond.py --env reacher-hard-v0 --gin_config_files config/online/sac_cond_synther_dmc.gin --gin_params 'redq_sac.cond_top_frac = 0.25'
```

## <a name="Citing"></a>Citing PGR

```BibTeX
@inproceedings{wang2025prioritized,
  title={Prioritized Generative Replay},
  author={Renhao Wang and Kevin Frans and Pieter Abbeel and Sergey Levine and Alexei A Efros},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=5IkDAfabuo}
}
```

## License and Acknowledgements

This codebase inherits all licenses from the public release of [SynthER](https://github.com/conglu1997/SynthER).
