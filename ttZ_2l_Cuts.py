def find_good_jet_indices(jet_pt,jet_jvt,
                         #,jet_eta # uncomment this when you want to impose stricter requirements on jets
                         ): # get indices of good jets
    good_jets = [] # list to hold whether jets are good
    for i in range(len(jet_pt)): # loop over jets
        if jet_pt[i]>25000: # paper: "Jets are accepted if they fulfill the requirements pT > 25 GeV"
            # paper: jets with pT < 60 GeV and |η| < 2.4 are required to satisfy pileup rejection criteria (JVT)
            if jet_pt[i]<60000: # extra requirements for pt < 60 GeV and |η|<2.4
            #if jet_pt[i]<60000 and abs(jet_eta[i])<2.4: # extra requirements for pt < 60 GeV and |η|<2.4
                if jet_jvt[i]<0.59: 
                    good_jets.append(0) # if jvt<0.59, this isn't a good jet
                    continue # move onto next jet
            good_jets.append(1) # append good jet if gets here
            continue # move onto next jet
        good_jets.append(0) # if pt<25000, this isn't a good jet
        
    string_ints = [str(i) for i in good_jets] # Convert each integer to a string
    str_of_ints = "".join(string_ints) # Combine each string
    return str_of_ints # return list of whether jets are good

# calculate number of good jets
def calc_goodjet_n(good_jets): # jet indices
    return good_jets.count('1') # count number of 1 in good jets list

# selection on number of good jets
def select_good_jet_n(good_jets):
    return good_jets.count('1')<5 # throw away if fewer than 5 good jets


# calculate number of (good) b-jets
def calc_bjet_n(jet_MV2c10,good_jets): # MV2c10 scores and jet indices
    bjet_n = 0 # start counter for number of b-jets
    good_jets_indices = [i for i,b in enumerate(good_jets) if b=='1'] # get the good jets
    for i in good_jets_indices: # loop over good jets
        if jet_MV2c10[i]>0.6459: # for 77% b-tag efficiency from https://cds.cern.ch/record/2160731/files/ATL-PHYS-PUB-2016-012.pdf Table 2                           
            bjet_n+=1 # increment the counter for the number of b-jets
    return bjet_n # return the number of b-jets

# selection on number of b-jets
def select_bjet_n(bjet_n):
    # throw away if fewer than 1 b-jets
    return bjet_n<1


# selection on nth good lepton
def select_leptons(mll, lep_ptcone30, lep_pt, lep_type, lep_etcone20, lep_tracksigd0pvunbiased, lep_eta, 
                   lep_charge
                   #,lep_z0 # uncomment this to apply stricter requirements
                  ): # variables for good lepton
    
    # paper: "invariant mass of the lepton pair is required to be in the Z boson mass window, |mll − mZ| < 10 GeV"
    if (mll < 81.12) or (mll > 101.12): return True
    
    # paper: "total sum of [...] transverse momenta in a surrounding cone [...] is required to be less than 6% of [...] pT"
    if lep_ptcone30[0]>0.06*lep_pt[0] or lep_ptcone30[1]>0.06*lep_pt[1]: return True # bad lepton if ptcone>6%pt
    
    # paper: "sum of [...] transverse energies [...] within a cone of size ∆Rη = 0.2 around any electron [...] is required to be less than 6% of [...] pT"
    if lep_type[0]==11 and lep_etcone20[0]>0.06*lep_pt[0]: return True # bad electron if etcone>6%pt
    if lep_type[1]==11 and lep_etcone20[1]>0.06*lep_pt[1]: return True # bad electron if etcone>6%pt
    
    # paper: "significance of [...] d0 is required to satisfy |d0|/σ(d0) < 5 for electrons and |d0|/σ(d0) < 3 for muons"
    if lep_type[0]==13 and lep_tracksigd0pvunbiased[0]>3: return True # bad muon if σ(d0)>3
    if lep_type[1]==13 and lep_tracksigd0pvunbiased[1]>3: return True # bad muon if σ(d0)>3
    if lep_tracksigd0pvunbiased[0]>5 or lep_tracksigd0pvunbiased[1]>5: return True # bad electron if σ(d0)>5
    
    # paper Table 2: pT (leading lepton) > 30 GeV and pT (subleading lepton) > 15 GeV
    if lep_pt[0]<30000 or lep_pt[1]<15000: return True # minimum pt requirments on leptons
    
    # paper: "Muons are required to have |η| < 2.5"
    if abs(lep_eta[0])>2.5 or abs(lep_eta[1])>2.5: return True # bad lepton if |η|>2.5
    
    # paper: "Opposite-sign"
    if lep_charge[0]+lep_charge[1]!=0: return True # throw away when charges don't add to 0
    
    # paper: "longitudinal impact parameter [...], z0, is required to satisfy |z0 sinθ| < 0.5 mm"
    #theta_i = 2*np.arctan(np.exp(-lep_eta[0])) # calculate theta angle
    #if abs(lep_z0[0]*np.sin(theta_i))>0.5: return True # bad lepton if z0*sinθ > 0.5mm
    #theta_i = 2*np.arctan(np.exp(-lep_eta[1])) # calculate theta angle
    #if abs(lep_z0[1]*np.sin(theta_i))>0.5: return True # bad lepton if z0*sinθ > 0.5mm
    
    return False # don't throw away event if gets here
