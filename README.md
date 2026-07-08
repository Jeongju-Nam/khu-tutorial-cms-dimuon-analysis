# Dimuon Analysis

## Exercises

- `ex01.py` is the scalar, event-by-event version. It is easier to read, but it
  stores every selected dimuon mass in a Python list before plotting.
- `ex02.py` is the chunked/vectorized version. It uses Awkward Array and NumPy to
  process many events at a time, accumulates fixed-bin histogram counts, and is
  the faster version for full samples.

Both scripts read the `Events` tree from:

```text
/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root
```

They select events with at least two muons, require the first two muons to have
opposite charge and `pT > 3 GeV`, then plot the dimuon invariant-mass spectrum.

## Recipes

### Install dependencies

```bash
source /store/sw/miniconda3/etc/profile.d/mamba.sh
mamba create -p ./dimuon-env python=3.12 uproot awkward numpy matplotlib tqdm
mamba activate ./dimuon-env
```

### Activate environment

```bash
source /store/sw/miniconda3/etc/profile.d/mamba.sh
mamba activate ./dimuon-env
```

### Run a quick smoke test

Use a small event count to confirm the environment and input file are available.

```bash
python ./ex01.py --num-events 100
python ./ex02.py --num-events 100
```

### Run the scalar exercise

```bash
python ./ex01.py
```

This writes `ex01.png`. Use `--num-events` to limit the number of events:

```bash
python ./ex01.py --num-events 10000
```

### Run the vectorized exercise

```bash
python ./ex02.py
```

This writes `ex02.png`. By default, `ex02.py` processes chunks of 16384 events.
Use `--step-size` to change the chunk size:

```bash
python ./ex02.py --num-events 100000 --step-size 32768
```

### Check outputs

```bash
ls -lh ex01.png ex02.png
```
