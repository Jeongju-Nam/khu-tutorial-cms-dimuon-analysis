from pathlib import Path

import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURE_DIRECTORY = Path(__file__).resolve().parent/"figures"
FIGURE_DIRECTORY.mkdir(parents = True, exist_ok = True)

ROOT_PATH = "/store/hep/users/slowmoyang/NanoAODRun1Examples/Run2012BC_DoubleMuParked_Muons.root"

MAXIMUM_EVENTS = 10_000

MUON_PT_MINIMUM = 3

NUMBER_OF_EVENTS_TO_PRINT = 10

# The width of the line that separates the printed events
SEPARATOR_WIDTH = 70

# The required branches
BRANCHES = [
    "nMuon",
    "Muon_charge",
    "Muon_pt",
    "Muon_eta",
    "Muon_phi",
    "Muon_mass"
]


def dimuon_mass():
    print()
    with uproot.open(ROOT_PATH) as root_file:
        tree = root_file["Events"]

        print(f"{'TTree name':<24}: {tree.name:>9}")
        print(f"{'Total entries in TTree':<24}: {tree.num_entries:>9}")

        events = tree.arrays(BRANCHES, entry_stop = MAXIMUM_EVENTS, library = "ak")

    events = ak.with_field(events, np.arange(len(events)), "entry")

    # 1st condition: events with two or more muons
    events_two_muons = events[events["nMuon"] >= 2]

    # 2nd condition: events where the first two muons have opposite charges
    opposite_charge_mask = (
        events_two_muons["Muon_charge"][:, 0] != events_two_muons["Muon_charge"][:, 1]
    )
    events_opposite_charge = events_two_muons[opposite_charge_mask]

    # 3rd condition: events where the first two muons both have pT >= 3 GeV
    pt_minimum_mask = (
        (events_opposite_charge["Muon_pt"][:, 0] >= MUON_PT_MINIMUM) &
        (events_opposite_charge["Muon_pt"][:, 1] >= MUON_PT_MINIMUM)
    )
    selected_events = events_opposite_charge[pt_minimum_mask]

    pt_1 = selected_events["Muon_pt"][:, 0]
    eta_1 = selected_events["Muon_eta"][:, 0]
    phi_1 = selected_events["Muon_phi"][:, 0]
    mass_1 = selected_events["Muon_mass"][:, 0]

    pt_2 = selected_events["Muon_pt"][:, 1]
    eta_2 = selected_events["Muon_eta"][:, 1]
    phi_2 = selected_events["Muon_phi"][:, 1]
    mass_2 = selected_events["Muon_mass"][:, 1]

    # Four-momentum of muon 1
    px_1 = pt_1*np.cos(phi_1)
    py_1 = pt_1*np.sin(phi_1)
    pz_1 = pt_1*np.sinh(eta_1)
    energy_1 = np.sqrt(px_1**2 + py_1**2 + pz_1**2 + mass_1**2)

    # Four-momentum of muon 2
    px_2 = pt_2*np.cos(phi_2)
    py_2 = pt_2*np.sin(phi_2)
    pz_2 = pt_2*np.sinh(eta_2)
    energy_2 = np.sqrt(px_2**2 + py_2**2 + pz_2**2 + mass_2**2)

    # Total four-momentum
    dimuon_px = px_1 + px_2
    dimuon_py = py_1 + py_2
    dimuon_pz = pz_1 + pz_2
    dimuon_energy = energy_1 + energy_2

    dimuon_mass_squared = dimuon_energy**2 - (dimuon_px**2 + dimuon_py**2 + dimuon_pz**2)
    dimuon_mass_squared = ak.where(dimuon_mass_squared >= 0, dimuon_mass_squared, 0)
    dimuon_masses = np.sqrt(dimuon_mass_squared)

    selected_events = ak.with_field(selected_events, dimuon_masses, "dimuon_mass")

    n_read = len(events)

    print()
    print("Cut flow")
    print(f"{'':<24}{'events':>9}{'fraction':>11}")
    print(f"{'Entries read':<24}{n_read:>9}{100:>10.1f}%")
    print(
        f"{'Two or more muons':<24}{len(events_two_muons):>9}"
        f"{100*len(events_two_muons)/n_read:>10.1f}%"
    )
    print(
        f"{'Opposite charges':<24}{len(events_opposite_charge):>9}"
        f"{100*len(events_opposite_charge)/n_read:>10.1f}%"
    )
    print(
        f"{'Both pT >= 3 GeV':<24}{len(selected_events):>9}"
        f"{100*len(selected_events)/n_read:>10.1f}%"
    )

    number_to_print = min(NUMBER_OF_EVENTS_TO_PRINT, len(selected_events))

    print()
    print("Selected event examples")
    for event in selected_events[:number_to_print]:
        print("-"*SEPARATOR_WIDTH)
        print(f"{'Entry':<24}: {int(event['entry']):>9}")
        print(f"{'nMuon':<24}: {int(event['nMuon']):>9}")
        print(f"{'Muon charges':<24}: {str(ak.to_list(event['Muon_charge'])):>9}")
        print(f"{'Muon pT [GeV]':<24}: {str(ak.to_list(event['Muon_pt'])):>9}")
        print(f"{'Dimuon mass [GeV]':<24}: {float(event['dimuon_mass']):>9.3f}")

    dimuon_masses_numpy = ak.to_numpy(dimuon_masses)

    fig, ax = plt.subplots(figsize = (8, 6))

    ax.hist(dimuon_masses_numpy, bins = 100, range = (0, 150), histtype = "step")
    ax.set_xlabel("Dimuon invariant mass [GeV]")
    ax.set_ylabel("Number of dimuon candidates")
    ax.set_yscale("log")
    ax.grid(True, linestyle = ":", alpha = 0.5)
    ax.tick_params(direction = "in", top = True, right = True)

    fig.tight_layout()
    fig.savefig(FIGURE_DIRECTORY/f"{Path(__file__).stem}.png", bbox_inches = "tight")
    plt.close(fig)


if __name__ == "__main__":
    dimuon_mass()
