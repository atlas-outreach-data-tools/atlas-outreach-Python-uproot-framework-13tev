# Cut on number of photons
# paper: "The data used in this channel are selected using a diphoton trigger, which requires two clusters"
def cut_photon_n (photon_n):
# want to discard any events where photon_n does not equal 2
    # exclamation mark (!) means "not"
    # so != means "not equal to"
    return photon_n != 2

# Cut on pseudorapidity outside the fiducial region
# paper: "Photon candidates are reconstructed in the fiducial region |η| < 2.37"
def cut_photon_eta_fiducial(photon_eta):
# want to discard any events where modulus of photon_eta > 2.37
    return photon_eta[0] > 2.37 or photon_eta[1] > 2.37 or photon_eta[0] < -2.37 or photon_eta[1] < -2.37

# Cut on pseudorapidity in barrel/end-cap transition region
# paper: "excluding the calorimeter barrel/end-cap transition region 1.37 < |η| < 1.52"
def cut_photon_eta_transition(photon_eta):
# want to discard events where modulus of photon_eta between 1.37 and 1.52
    if photon_eta[0] < 1.52 and photon_eta[0] > 1.37: return True
    elif photon_eta[1] < 1.52 and photon_eta[1] > 1.37: return True
    elif photon_eta[0] > -1.52 and photon_eta[0] < -1.37: return True
    elif photon_eta[1] < -1.37 and photon_eta[1] > -1.52: return True
    else: return False

# Cut on Transverse momentum
# paper: "The leading (sub-leading) photon candidate is required to have ET > 40 GeV (30 GeV)"
def cut_photon_pt(photon_pt):
# want to discard any events where photon_pt[0] < 40000 MeV or photon_pt[1] < 30000 MeV
    # first lepton is [0], 2nd lepton is [1] etc
    return photon_pt[0] < 40000 or photon_pt[1] < 30000

# Cut on photon reconstruction
# paper: "Photon candidates are required to pass identification criteria"
def cut_photon_reconstruction(photon_isTightID):
# want to discard events where it is false for one or both photons
    return photon_isTightID[0] == False or photon_isTightID[1] == False

# Cut on energy isolation
# paper: "Photon candidates are required to have an isolation transverse energy of less than 4 GeV"
def cut_isolation_et(photon_etcone20):
# want to discard events where isolation eT > 4000 MeV
    return photon_etcone20[0] > 4000 or photon_etcone20[1] > 4000

# Cut on reconstructed invariant mass lower limit
# paper: "in the diphoton invariant mass range between 100 GeV and 160 GeV"
def cut_mass_lower(myy):
# want to discard minimum invariant reconstructed mass < 100 GeV
    return myy < 100

# Cut on reconstructed invariant mass upper limit
# paper: "in the diphoton invariant mass range between 100 GeV and 160 GeV"
def cut_mass_upper(myy):
# want to discard maximum invariant reconstructed mass > 160 GeV
    return myy > 160
