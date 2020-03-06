# cut on number of leptons
def cut_n_lep(lep_n):
    # exclamation mark (!) means "not"
    # so != means "not equal to"
    # return when number of leptons is not equal to 4
    return lep_n != 4

# cut on lepton charge
def cut_lep_charge(lep_charge):
    # return when sum of lepton charges is not equal to 0
    # first lepton is [0], 2nd lepton is [1] etc
    return lep_charge[0] + lep_charge[1] + lep_charge[2] + lep_charge[3] != 0

#cut on transverse momentum of the leptons
def cut_lep_pt_012(lep_pt):
#want to throw away any events where lep_pt[1] < 15000
#want to throw away any events where lep_pt[2] < 10000
    return lep_pt[1] < 15000 or lep_pt[2] < 10000

# cut on lepton type
def cut_lep_type(lep_type):
# for an electron lep_type is 11
# for a muon lep_type is 13
    sum_lep_type = lep_type[0] + lep_type[1] + lep_type[2] + lep_type[3]
    return (sum_lep_type != 44) and (sum_lep_type != 48) and (sum_lep_type != 52)

# cut on lepton momentum isolation
def cut_lep_ptcone(lep_ptcone30,lep_pt):
    return lep_ptcone30[0]/lep_pt[0] > 0.3 or lep_ptcone30[1]/lep_pt[1] > 0.3 or lep_ptcone30[2]/lep_pt[2] > 0.3 or lep_ptcone30[3]/lep_pt[3] > 0.3

# cut on lepton energy isolation
def cut_lep_etcone(lep_etcone20,lep_pt):
    return lep_etcone20[0]/lep_pt[0] > 0.3 or lep_etcone20[1]/lep_pt[1] > 0.3 or lep_etcone20[2]/lep_pt[2] > 0.3 or lep_etcone20[3]/lep_pt[3] > 0.3
