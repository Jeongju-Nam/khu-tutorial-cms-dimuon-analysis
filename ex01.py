"""Build a dimuon invariant-mass spectrum one event at a time.

This exercise reads the ``Events`` tree from the CMS Open Data NanoAOD Run 2012
DoubleMuParked sample and keeps events whose first two muons have opposite
charge and transverse momentum above 3 GeV. It computes the invariant mass from
the two muon four-vectors, fills a Python list, and writes the spectrum to
``ex01.png``.

This file is intentionally written in a direct, event-by-event style. That makes
it slower than a vectorized analysis, but it is useful for learning because each
line corresponds to one familiar analysis step:

1. Open the ROOT file.
2. Read one collision event.
3. Apply simple event-selection cuts.
4. Convert each selected muon from detector coordinates to a four-vector.
5. Add the two four-vectors and compute the dimuon invariant mass.
6. Fill and plot the mass spectrum.

Some Python notes for new readers:

- A function is a named block of reusable code. For example,
  ``compute_mass(...)`` is used every time we need an invariant mass.
- A list, such as ``dimuon_mass_list``, stores many values in order.
- ``continue`` skips the rest of the current event and moves to the next event.
- Square brackets select data from a container. For example,
  ``event["Muon_pt"][0]`` means "the transverse momentum of the first muon".

Reference analysis:
https://github.com/cms-opendata-analyses/NanoAODRun1Examples/blob/main/dimuon_2012/dimuonSpectrum2012_eospublic.py
"""
import time
import argparse
import uproot
import math
import tqdm
import matplotlib.pyplot as plt


def compute_px_py_pz_energy(pt, eta, phi, mass):
    """Convert one particle's kinematics into momentum components and energy.

    The NanoAOD file stores muons using variables that are natural for a
    collider detector:

    - pt: transverse momentum, the momentum perpendicular to the beam pipe
    - eta: pseudorapidity, related to the polar angle from the beam pipe
    - phi: azimuthal angle around the beam pipe
    - mass: the reconstructed muon mass

    For invariant-mass calculations it is convenient to use the Cartesian
    four-vector (px, py, pz, E). The formulas below do that conversion.
    """
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)

    # Relativistic energy relation in units where c = 1:
    # E^2 = p^2 + m^2, with p^2 = px^2 + py^2 + pz^2.
    momentum2 = px**2 + py**2 + pz**2
    energy = math.sqrt(momentum2 + mass**2)

    return px, py, pz, energy


def compute_mass(px, py, pz, energy):
    """Compute invariant mass from a four-vector.

    For two muons, first add their four-vectors component by component. The
    invariant mass of the pair is then

        m^2 = E^2 - (px^2 + py^2 + pz^2)

    Small negative values can appear from floating-point roundoff, so ``max``
    protects the square root from tiny unphysical negatives.
    """
    mass2 = energy**2 - (px**2 + py**2 + pz**2)
    mass = math.sqrt(max(mass2, 0.0))
    return mass


parser = argparse.ArgumentParser(description="Dimuon mass spectrum analysis")
parser.add_argument("-n", "--num-events", type=int, default=None, help="Number of events to process (default: all events)")
args = parser.parse_args()


path = "/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root"
file = uproot.open(path)

# ROOT files contain trees, which are table-like collections of event data.
# In NanoAOD, the main physics tree is named "Events".
tree = file['Events']

# ``iterate`` reads the tree in pieces. ``library='np'`` asks uproot to return
# NumPy arrays, a common Python format for numerical data.
tree_iterator = tree.iterate(step_size=1, library='np')

# If the user gives --num-events, process only that many events. Otherwise use
# all entries in the tree.
num_events = args.num_events or tree.num_entries

# We append one number to this list for every selected dimuon candidate.
dimuon_mass_list = []

analysis_start = time.time()

for chunk in tqdm.tqdm(tree.iterate(entry_stop=num_events, step_size=1, library='np'), total=num_events):
    # With step_size=1, each chunk contains one event. Uproot still returns
    # arrays with one entry, so value[0] strips off that one-event wrapper.
    event = {key: value[0] for key, value in chunk.items()}

    # Require at least two reconstructed muons so that [0] and [1] exist below.
    if event['nMuon'] < 2:
        continue

    # Keep only opposite-charge pairs. Same-charge pairs are less useful for
    # this resonance spectrum because the signal decays produce mu+ mu-.
    if event['Muon_charge'][0] == event['Muon_charge'][1]:
        continue

    # Require both muons to have transverse momentum above 3 GeV. Low-pT muons
    # are harder to reconstruct reliably and are outside this exercise's cut.
    if (event['Muon_pt'][0] <= 3) or (event['Muon_pt'][1] <= 3):
        continue

    # This exercise intentionally uses only the first two muons in each event.
    # More advanced analyses would consider all possible muon pairs.
    px_0, py_0, pz_0, energy_0 = compute_px_py_pz_energy(
        pt=event['Muon_pt'][0],
        eta=event['Muon_eta'][0],
        phi=event['Muon_phi'][0],
        mass=event['Muon_mass'][0],
    )

    px_1, py_1, pz_1, energy_1 = compute_px_py_pz_energy(
        pt=event['Muon_pt'][1],
        eta=event['Muon_eta'][1],
        phi=event['Muon_phi'][1],
        mass=event['Muon_mass'][1],
    )

    px = px_0 + px_1
    py = py_0 + py_1
    pz = pz_0 + pz_1
    energy = energy_0 + energy_1

    # Store the invariant mass of this selected two-muon system.
    mass = compute_mass(px, py, pz, energy)
    dimuon_mass_list.append(mass)

analysis_elapsed_time = time.time() - analysis_start
print(f"Analysis time: {analysis_elapsed_time:.2f} seconds")

plotting_start = time.time()

fig, ax = plt.subplots()

# Draw a histogram: each selected event contributes one dimuon mass. The range
# covers light resonances through the Z boson, and many bins make narrow peaks
# visible on the logarithmic x axis.
ax.hist(dimuon_mass_list, bins=30_000, range=(0.25, 300.0))

# Log scales make both low-mass and high-mass structures visible on one plot.
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(0.25, 300.0)

# Axis titles. The r"..." strings are "raw strings", useful for LaTeX-style
# labels because backslashes are kept as normal characters.
ax.set_xlabel(r"$m_{\mu\mu}$ (GeV)", fontsize=14)
ax.set_ylabel(r"$N_{\mathrm{Events}}$", fontsize=14)

ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

# Vertical reference lines mark known resonances that can decay to muon pairs.
ax.axvline(0.55, color="blue", linestyle="--", label=r"$\eta$")
ax.axvline(0.77, color="orange", linestyle="--", label=r"$\rho,\omega$")
ax.axvline(1.2, color="green", linestyle="--", label=r"$\phi$")
ax.axvline(4.4, color="red", linestyle="--", label=r"$J/\psi$")
ax.axvline(4.6, color="purple", linestyle="--", label=r"$\psi'$")
ax.axvline(12.0, color="brown", linestyle="--", label=r"$Y(1,2,3S)$")
ax.axvline(91.0, color="pink", linestyle="--", label=r"$Z$")

ax.legend(
    loc="upper right",
    fontsize=9,
    frameon=False,
    ncol=2,
)

plt.tight_layout()
plt.savefig('./ex01.png')

plotting_elapsed_time = time.time() - plotting_start
print(f"Plotting time: {plotting_elapsed_time:.2f} seconds")
