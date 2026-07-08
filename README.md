# Dimuon Analysis

## Recipes

### Install dependencies
```bash
source /store/sw/miniconda3/etc/profile.d/mamba.sh
mamba create -p ./dimuon-env python=3.12 uproot numpy matplotlib tqdm
mamba activate ./dimuon-env
```

### Activate environment
```bash
source /store/sw/miniconda3/etc/profile.d/mamba.sh
mamba activate ./dimuon-env
```

### Run the code
```bash
python ./ex01.py
python ./ex02.py
```

`ex02.py` is a vectorized successor of `ex01.py`. It processes events with
`step_size=1024`, computes the dimuon analysis per chunk, and writes `ex02.png`.
