"""
https://github.com/cms-opendata-analyses/NanoAODRun1Examples/blob/main/dimuon_2012/dimuonSpectrum2012_eospublic.py
"""
import uproot
import math
import tqdm
import matplotlib.pyplot as plt

def compute_px_py_pz_energy(pt, eta, phi, mass):
    px = pt * math.cos(phi)
    py = pt * math.sin(phi)
    pz = pt * math.sinh(eta)

    momentum2 = px**2 + py**2 + pz**2
    energy = math.sqrt(momentum2 + mass**2)

    return px, py, pz, energy

def compute_mass(px, py, pz, energy):
    mass2 = energy**2 - (px**2 + py**2 + pz**2)
    mass = math.sqrt(max(mass2, 0.0))
    return mass

path = "/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root"
file = uproot.open(path)

tree = file['Events']
tree_iterator = tree.iterate(step_size=1, library='np')

max_events = min(10_000, tree.num_entries)

dimuon_mass_list = []

for chunk in tqdm.tqdm(tree.iterate(entry_stop=max_events, step_size=1, library='np'), total=max_events):
    event = {key: value[0] for key, value in chunk.items()}

    # Select events with at least two muons
    if event['nMuon'] < 2:
        continue

    # Select events with two muons of opposite charge and pT> 3 GeV
    if event['Muon_charge'][0] == event['Muon_charge'][1]:
        continue

    if (event['Muon_pt'][0] <= 3) or (event['Muon_pt'][1] <= 3):
        continue

    px_0, py_0, pz_0, energy_0 = compute_px_py_pz_energy(
        pt=event['Muon_pt'][0],
        eta=event['Muon_eta'][0],
        phi=event['Muon_eta'][0],
        mass=event['Muon_mass'][0],
    )

    px_1, py_1, pz_1, energy_1 = compute_px_py_pz_energy(
        pt=event['Muon_pt'][1],
        eta=event['Muon_eta'][1],
        phi=event['Muon_eta'][1],
        mass=event['Muon_mass'][1],
    )

    px = px_0 + px_1
    py = py_0 + py_1
    pz = pz_0 + pz_1
    energy = energy_0 + energy_1

    mass = compute_mass(px, py, pz, energy)
    dimuon_mass_list.append(mass)

fig, ax = plt.subplots()

ax.hist(dimuon_mass_list, bins=30_000, range=(0.25, 300.0))

# Log scales
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlim(0.25, 300.0)

# Axis titles
ax.set_xlabel(r"$m_{\mu\mu}$ (GeV)", fontsize=14)
ax.set_ylabel(r"$N_{\mathrm{Events}}$", fontsize=14)

ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)

# Resonance vertical lines with axvline labels, without loop

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
