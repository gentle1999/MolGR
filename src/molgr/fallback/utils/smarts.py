from __future__ import annotations

from openbabel import pybel


def _smarts(pattern: str) -> pybel.Smarts:
    return pybel.Smarts(pattern)


# The first two atoms identify the suspect single bond in the seed graph.
LONG_SEED_BONDS = (
    _smarts("[#14D4v4]-[#6D2v3]=[#8]"),
    _smarts("[#15D4v4]-[#15D4v4]"),
)

# Each row matches the two atoms independently before a bond exists.
OCTAHEDRAL_PHOSPHORUS_RULES = ((_smarts("[#15D5v5]"), _smarts("[#8v1,#6v3]")),)
PRIMARY_CONNECTION_RULES = (
    (
        _smarts("[Nv0,Cv1,Nv3,Clv1,Clv2,Clv3,Brv1,Brv2,Brv3,Iv1,Iv2,Iv3,Asv3,Sev2]"),
        _smarts("[Hv0,Bv2,Bv3,Cv0,Cv1,Cv2,Cv3,Nv1,Nv2,Ov0,Ov1,Clv0,Siv3,Pv2,Sv0,Sv1,Brv0,Iv0]"),
    ),
    (_smarts("[Pv2]"), _smarts("[Cv0,Cv1,Cv2,Cv3,Nv2]")),
)
SECONDARY_CONNECTION_RULES = (
    (_smarts("[Ov1]"), _smarts("[Cv3]")),
    (_smarts("[#34v3,#53v4,#53v5]"), _smarts("[Ov0,Ov1]")),
)
# Aggregate queries kept for callers that inspect candidate atoms.
PREPROCESS_ACCEPT = PRIMARY_CONNECTION_RULES[0][1]
PREPROCESS_DONATE = _smarts("[Nv0,Cv1,Nv3,Pv2,Clv1,Clv2,Clv3,Brv1,Brv2,Brv3,Iv1,Iv2,Iv3,Asv3,Sev2]")
PREPROCESS_ADDITIONAL_DONORS = _smarts("[Ov1,#34v3,#53v4,#53v5]")

PRE_CLEAN_HYPERVALENT = _smarts("[Cv5,Nv5,Pv5,Siv5]=,#[*]")
# Open Babel may perceive sulfur in an aromatic heterocycle as ``s``. Match by
# atomic number so an aromatic S=O bond still gets normalized to S-O before
# formal charge inference.
PRE_CLEAN_HYPER_PI_BOND = _smarts("[#16,#15,#33,#9,#17,#35,#53]=,#[*]")
PRE_CLEAN_BCP_RING_5 = _smarts("[#6]1([#6]2)([#6]3)[#7]23[#6]1")
PRE_CLEAN_BCP_RING_4 = _smarts("[#6]1([#6]2)[#7]2[#6]1")

ELIM_HIGH_POSITIVE = _smarts("[*+1,*+2,*+3]-[Ov1+0,Nv2+0,Sv1+0]")
ELIM_CARBOXYL = _smarts("[Ov1+0]-C=O")
ELIM_NNN_NEGATIVE = _smarts("[#7v1+0]-[#7v2+0]-[#7v1+0]")
ELIM_NNN_POSITIVE = _smarts("[#7v3+0]-[#7v2+0]-[#7v3+0]")
ELIM_1_3_DIPOLE_POSTIVE = _smarts("[*-1]-,=[N+0,O+0,S+0,P+0]-,=[*]")
ELIM_POSITIVE_N = _smarts("[Nv3+0]=[Nv2+0]")
ELIM_POSITIVE_C_H = _smarts("[#6v3+0,#6v2+0,#1v0+0]")
ELIM_POSITIVE_DIPOLE = _smarts("[*]-,=[Nv3,Ov2,Pv3,Sv2]")
ELIM_NEGATIVE_F = _smarts("[#9v0+0]")
ELIM_NEGATIVE_O = _smarts("[#8v0+0]")
ELIM_NEGATIVE_O_1 = _smarts("[#8v1+0]")
ELIM_NEGATIVE_CL = _smarts("[#17v0+0]")
ELIM_NEGATIVE_N = _smarts("[#7v0+0]")
ELIM_NEGATIVE_N_1 = _smarts("[#7v1+0]")
ELIM_NEGATIVE_N_2 = _smarts("[#7v2+0]")
ELIM_NEGATIVE_BR = _smarts("[#35v0+0]")
ELIM_NEGATIVE_I = _smarts("[#53v0+0]")
ELIM_NEGATIVE_S = _smarts("[#16v0+0]")
ELIM_NEGATIVE_S_1 = _smarts("[#16v1+0]")
ELIM_NEGATIVE_SE = _smarts("[#34v0+0]")
ELIM_NEGATIVE_SE_1 = _smarts("[#34v1+0]")
ELIM_NEGATIVE_P = _smarts("[#15v0+0]")
ELIM_NEGATIVE_P_1 = _smarts("[#15v1+0]")
ELIM_NEGATIVE_P_2 = _smarts("[#15v2+0]")
ELIM_NEGATIVE_B = _smarts("[#5v0+0]")
ELIM_NEGATIVE_B_1 = _smarts("[#5v1+0]")
ELIM_NEGATIVE_B_2 = _smarts("[#5v2+0]")
ELIM_NEGATIVE_C_V3 = _smarts("[#6v3+0]")
ELIM_NEGATIVE_H = _smarts("[#1v0+0]")
ELIM_NEGATIVE_C_LOW = _smarts("[#6v2+0,#6v1+0,#6v0+0]")
ELIM_NEGATIVE_CP = _smarts(
    "[#6v3+0,#6v4+0]1=[#6v3+0,#6v4+0]-[#6v3+0,#6v4+0]=[#6v3+0,#6v4+0]-[#6v2+0,#6v3+0]-1"
)

CLEAN_CARBENE_NEIGHBOR_UNSAT = _smarts("[*]-[*]=[*]")
CLEAN_POSSIBLE_1_3_DIPOLE = _smarts("[*]-,=[N,O,P,S]-,=[*]")
CLEAN_1_4_RADICALS = _smarts("[*]-[*]=[*]-[*]")
CLEAN_1_6_RADICALS = _smarts("[*]-[*]=[*]-[*]=[*]-[*]")
CLEAN_RESONANCE_0 = _smarts("[*-]-[*]=[*]~[*+]")
CLEAN_RESONANCE_1 = _smarts("[*-]=[*+]=[*+0]")
CLEAN_RESONANCE_2 = _smarts("[#8]=[#6](-[!-])-[*]=[*]-[#7-,#6-]")
CLEAN_RESONANCE_3 = _smarts("[#7v2+]=[*]-[*]=[*]-[#8-]")
CLEAN_RESONANCE_4 = _smarts("[#7+,#8+]=[*]-[#6-,#7-,#8-]")
CLEAN_RESONANCE_5 = _smarts("[#7+0,#8+0,#16+0]=[#6+0]-[#6-,#7-]")
CLEAN_RESONANCE_6 = _smarts("[#6]=[#6]=[#6-,#7-]")
CLEAN_RESONANCE_7 = _smarts("[*-]1-,:[*](=,:[*])-,:[*]=,:[*]-,:[*]=,:[*]1")
CLEAN_RESONANCE_8 = _smarts("[*-]1-,:[*]=,:[*]-,:[*](=,:[*])-,:[*]=,:[*]1")
CLEAN_RESONANCE_9 = _smarts("[*+,*+2,*+3]-,=[*-,*-2,*-3]")
CLEAN_RESONANCE_10 = _smarts("[*]-[*]=,#[*]-[*]")
CLEAN_RESONANCE_11 = _smarts("[#7v3+0,#8v2+0,#16v2+0]-,=,:[*+1]")
CLEAN_RESONANCE_12 = _smarts("[#7v3+0,#8v2+0,#16v2+0]-,:[*]=,:[*]-,:[*+1]")
CLEAN_RESONANCE_13 = _smarts("[*-]:[*]=[#7+0,#8+0]")
CLEAN_RESONANCE_14 = _smarts("[#15-,#16-,#17-,#35-,#53-]#[#7+1,#8+1,#16+1]")
CLEAN_RESONANCE_16 = _smarts("[*-1]-,:[*]=,:[*]-,:[*]=,:[*+1]")
CLEAN_RESONANCE_17 = _smarts("[*-]-[*R]=[*R]=[*R]")
CLEAN_RESONANCE_18 = _smarts("[*]-[#7+0]=[#7+0]-[#7+0]")


BREAK_DEFORMED_ENE_A = _smarts("[*]~[*+0]=,:[*+0]~[*]")
BREAK_DEFORMED_ENE_B = _smarts("[*]~[*+0](=,:[*+0])~[*]")
BREAK_ONE_BOND_MULTIPLE = _smarts("[*+0]#,=[*+0]")
BREAK_ONE_BOND_CATION = _smarts("[#7+1,#15+1]=[*+0]")
BREAK_ONE_BOND_AROMATIC = _smarts("[*+0]:[*+0]")

RESONANCE_ONE_STEP = _smarts("[*]-,=,:[*]=,#,:[*]")
SCORING_CONJUGATION = _smarts("[*]=,#,:[*]-,:[*]=,#,:[*]")


__all__ = [
    "BREAK_DEFORMED_ENE_A",
    "BREAK_DEFORMED_ENE_B",
    "BREAK_ONE_BOND_AROMATIC",
    "BREAK_ONE_BOND_CATION",
    "BREAK_ONE_BOND_MULTIPLE",
    "CLEAN_CARBENE_NEIGHBOR_UNSAT",
    "CLEAN_1_4_RADICALS",
    "CLEAN_1_6_RADICALS",
    "CLEAN_POSSIBLE_1_3_DIPOLE",
    "CLEAN_RESONANCE_0",
    "CLEAN_RESONANCE_1",
    "CLEAN_RESONANCE_10",
    "CLEAN_RESONANCE_11",
    "CLEAN_RESONANCE_12",
    "CLEAN_RESONANCE_13",
    "CLEAN_RESONANCE_14",
    "CLEAN_RESONANCE_16",
    "CLEAN_RESONANCE_17",
    "CLEAN_RESONANCE_18",
    "CLEAN_RESONANCE_2",
    "CLEAN_RESONANCE_3",
    "CLEAN_RESONANCE_4",
    "CLEAN_RESONANCE_5",
    "CLEAN_RESONANCE_6",
    "CLEAN_RESONANCE_7",
    "CLEAN_RESONANCE_8",
    "CLEAN_RESONANCE_9",
    "ELIM_1_3_DIPOLE_POSTIVE",
    "ELIM_CARBOXYL",
    "ELIM_HIGH_POSITIVE",
    "ELIM_NEGATIVE_B",
    "ELIM_NEGATIVE_B_1",
    "ELIM_NEGATIVE_B_2",
    "ELIM_NEGATIVE_BR",
    "ELIM_NEGATIVE_C_LOW",
    "ELIM_NEGATIVE_C_V3",
    "ELIM_NEGATIVE_CL",
    "ELIM_NEGATIVE_F",
    "ELIM_NEGATIVE_H",
    "ELIM_NEGATIVE_I",
    "ELIM_NEGATIVE_N",
    "ELIM_NEGATIVE_N_1",
    "ELIM_NEGATIVE_N_2",
    "ELIM_NEGATIVE_O",
    "ELIM_NEGATIVE_O_1",
    "ELIM_NEGATIVE_P",
    "ELIM_NEGATIVE_P_1",
    "ELIM_NEGATIVE_P_2",
    "ELIM_NEGATIVE_S",
    "ELIM_NEGATIVE_S_1",
    "ELIM_NEGATIVE_SE",
    "ELIM_NEGATIVE_SE_1",
    "ELIM_NNN_NEGATIVE",
    "ELIM_NNN_POSITIVE",
    "ELIM_POSITIVE_C_H",
    "ELIM_POSITIVE_DIPOLE",
    "ELIM_POSITIVE_N",
    "PREPROCESS_ACCEPT",
    "PREPROCESS_ADDITIONAL_DONORS",
    "PREPROCESS_DONATE",
    "LONG_SEED_BONDS",
    "OCTAHEDRAL_PHOSPHORUS_RULES",
    "PRIMARY_CONNECTION_RULES",
    "SECONDARY_CONNECTION_RULES",
    "PRE_CLEAN_BCP_RING_4",
    "PRE_CLEAN_BCP_RING_5",
    "PRE_CLEAN_HYPERVALENT",
    "RESONANCE_ONE_STEP",
    "SCORING_CONJUGATION",
]
