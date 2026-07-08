"""
Vectorized successor of ex01.py.

Changes:
- process events in chunks with step_size=1024
- vectorize dimuon selection and mass calculation within each chunk
- use Muon_phi for px/py calculation
"""
import awkward as ak
import numpy as np
import tqdm
import uproot
import matplotlib.pyplot as plt


def compute_px_py_pz_energy(pt, eta, phi, mass):
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)

    momentum2 = px**2 + py**2 + pz**2
    energy = np.sqrt(momentum2 + mass**2)

    return px, py, pz, energy


def compute_mass(px, py, pz, energy):
    mass2 = energy**2 - (px**2 + py**2 + pz**2)
    mass = np.sqrt(ak.where(mass2 > 0.0, mass2, 0.0))
    return mass


path = "/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root"
file = uproot.open(path)

tree = file["Events"]

max_events = min(10_000, tree.num_entries)
step_size = 1024

dimuon_mass_list = []

branches = [
    "nMuon",
    "Muon_charge",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass",
]

with tqdm.tqdm(total=max_events) as progress:
    for chunk in tree.iterate(
        branches,
        entry_stop=max_events,
        step_size=step_size,
        library="ak",
    ):
        progress.update(len(chunk["nMuon"]))

        # Select events with at least two muons
        has_two_muons = chunk["nMuon"] >= 2
        selected = chunk[has_two_muons]

        if len(selected["nMuon"]) == 0:
            continue

        muon_charge_0 = selected["Muon_charge"][:, 0]
        muon_charge_1 = selected["Muon_charge"][:, 1]
        muon_pt_0 = selected["Muon_pt"][:, 0]
        muon_pt_1 = selected["Muon_pt"][:, 1]

        # Select events with two muons of opposite charge and pT > 3 GeV
        event_mask = (
            (muon_charge_0 != muon_charge_1)
            & (muon_pt_0 > 3)
            & (muon_pt_1 > 3)
        )
        selected = selected[event_mask]

        if len(selected["nMuon"]) == 0:
            continue

        px_0, py_0, pz_0, energy_0 = compute_px_py_pz_energy(
            pt=selected["Muon_pt"][:, 0],
            eta=selected["Muon_eta"][:, 0],
            phi=selected["Muon_phi"][:, 0],
            mass=selected["Muon_mass"][:, 0],
        )

        px_1, py_1, pz_1, energy_1 = compute_px_py_pz_energy(
            pt=selected["Muon_pt"][:, 1],
            eta=selected["Muon_eta"][:, 1],
            phi=selected["Muon_phi"][:, 1],
            mass=selected["Muon_mass"][:, 1],
        )

        px = px_0 + px_1
        py = py_0 + py_1
        pz = pz_0 + pz_1
        energy = energy_0 + energy_1

        mass = compute_mass(px, py, pz, energy)
        dimuon_mass_list.extend(ak.to_list(mass))

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
plt.savefig("./ex02.png")
