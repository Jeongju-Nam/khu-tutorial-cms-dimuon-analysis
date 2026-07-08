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

## Arrays used in this tutorial

### NumPy arrays

A NumPy array is a fast, fixed-shape container for numerical data. It is similar
to a Python list, but it stores values in a compact typed block of memory and
can apply math to all values at once:

```python
import numpy as np

pt = np.array([4.2, 18.0, 42.5])
passes_cut = pt > 3.0
```

In this tutorial, NumPy is useful when the data are regular: bin edges,
histogram counts, and per-muon calculations after selecting one value per event.
For example, `ex02.py` uses `np.cos`, `np.sin`, `np.sqrt`, `np.linspace`,
`np.zeros`, and `np.histogram`.

### Awkward arrays

An Awkward Array is an array library for nested or variable-length data. That is
important for collision events because each event can contain a different number
of muons:

```python
# Event 0 has two muons, event 1 has one, event 2 has three.
muon_pt = [[12.0, 8.5], [5.1], [31.0, 9.2, 4.8]]
```

This shape is awkward for plain NumPy because the rows do not all have the same
length. Awkward keeps that event structure while still allowing array-style
selection and math:

```python
selected = chunk[chunk["nMuon"] >= 2]
first_muon_pt = selected["Muon_pt"][:, 0]
```

In `ex02.py`, uproot returns chunks with `library="ak"`, so branches such as
`Muon_pt`, `Muon_eta`, and `Muon_charge` are Awkward arrays. After the event
selection produces one dimuon mass per selected event, `ak.to_numpy(mass)`
converts that regular result to NumPy for histogramming.

Use NumPy when the data are rectangular and numeric. Use Awkward when the data
are nested, jagged, or event-based with different numbers of objects per event.

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
