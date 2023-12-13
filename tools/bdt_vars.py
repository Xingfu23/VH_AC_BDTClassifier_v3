def converttoxml(dataset, datatype):
    dataset_forxml = []
    for item in dataset:
        name = item
        type = datatype
        dataset_forxml.append((name, type))
    return dataset_forxml

dataset = [
    "pho1_eta",
    "pho2_eta",
    "pho1_phi",
    "pho2_phi",
    "pho1_ptoM",
    "pho2_ptoM",
    "dipho_cosphi",
    "dipho_deltaeta",
    "met",
    "met_phi",
    "met_sumEt",
    "dphi_pho1_met",
    "dphi_pho2_met",
    "pt_balance",
    "njet",
    "max_jet_pt",
    "min_dphi_jet_met"
]

dataset_v2 = dataset + ["vtxprob", "sigmarv", "sigmawv", "max_phoId", "min_phoId"]

dataset_forxml = [
    ("pho1_eta", "F"),
    ("pho2_eta", "F"),
    ("pho1_phi", "F"),
    ("pho2_phi", "F"),
    ("pho1_ptoM", "F"),
    ("pho2_ptoM", "F"),
    ("dipho_cosphi", "F"),
    ("dipho_deltaeta", "F"),
    ("met", "F"),
    ("met_phi", "F"),
    ("met_sumEt", "F"),
    ("dphi_pho1_met", "F"),
    ("dphi_pho2_met", "F"),
    ("pt_balance", "F"),
    ("njet", "F"),
    ("max_jet_pt", "F"),
    ("min_dphi_jet_met", "F")
]

dataset_v2_forxml = converttoxml(dataset_v2, "F")