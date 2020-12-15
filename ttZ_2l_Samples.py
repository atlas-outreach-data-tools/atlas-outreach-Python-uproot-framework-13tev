samples = {

    'data': {
        'list' : [
                  #'data_A', # period A from data16
                  'data_B', # period B from data16
                  'data_C', # period C from data16
                  'data_D' # period D from data16
                 ] 
    },
    
    r'$t\bar{t}Z$' : { # ttZ(->ee) and ttZ(->μμ) signal (associated production of a top-quark pair with one vector boson)
        'list' : [
                  'ttee',
                  'ttmumu'
                 ]
    },

    r'$t\bar{t}$' : { # ttbar semileptonic / all-leptonic
        'list' : ['ttbar_lep']
    },
    
    'Z' : { # Z+jets (Events containing Z bosons with associated jets) 
        'list' : [
                  #'Zmumu_PTV0_70_CVetoBVeto', # Z->μμ + jets with 0 < pT(Z) < 70 GeV whilst vetoing c and b-jets
                  #'Zmumu_PTV0_70_CFilterBVeto', # Z->μμ + jets with 0 < pT(Z) < 70 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Zmumu_PTV0_70_BFilter', # Z->μμ + jets with 0 < pT(Z) < 70 GeV and a requirement for b-jets
                  #'Zmumu_PTV70_140_CVetoBVeto', # Z->μμ + jets with 70 < pT(Z) < 140 GeV whilst vetoing c and b-jets
                  #'Zmumu_PTV70_140_CFilterBVeto', # Z->μμ + jets with 70 < pT(Z) < 140 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Zmumu_PTV70_140_BFilter', # Z->μμ + jets with 70 < pT(Z) < 140 GeV and a requirement for b-jets
                  #'Zmumu_PTV140_280_CVetoBVeto', # Z->μμ + jets with 140 < pT(Z) < 280 GeV whilst vetoing c and b-jets
                  'Zmumu_PTV140_280_CFilterBVeto', # Z->μμ + jets with 140 < pT(Z) < 280 GeV and a requirement for c-jets, whilst vetoing b-jets
                  'Zmumu_PTV140_280_BFilter', # Z->μμ + jets with 140 < pT(Z) < 280 GeV and a requirement for b-jets
                  #'Zmumu_PTV280_500_CVetoBVeto', # Z->μμ + jets with 280 < pT(Z) < 500 GeV whilst vetoing c and b-jets
                  'Zmumu_PTV280_500_CFilterBVeto', # Z->μμ + jets with 280 < pT(Z) < 500 GeV and a requirement for c-jets, whilst vetoing b-jets
                  'Zmumu_PTV280_500_BFilter', # Z->μμ + jets with 280 < pT(Z) < 500 GeV and a requirement for b-jets
                  'Zmumu_PTV500_1000', # Z->μμ + jets, with 500 < pT(Z) < 1000 GeV 
                  #'Zmumu_PTV1000_E_CMS', # Z->μμ + jets with 1000 GeV < pT(Z) < centre-of-mass energy
                  #'Zee_PTV0_70_CVetoBVeto', # Z->ee + jets with 0 < pT(Z) < 70 GeV whilst vetoing c and b-jets
                  #'Zee_PTV0_70_CFilterBVeto', # Z->ee + jets with 0 < pT(Z) < 70 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Zee_PTV0_70_BFilter', # Z->ee + jets with 0 < pT(Z) < 70 GeV and a requirement for b-jets
                  #'Zmumu_PTV70_140_CVetoBVeto', # Z->ee + jets with 70 < pT(Z) < 140 GeV whilst vetoing c and b-jets
                  #'Zee_PTV70_140_CFilterBVeto', # Z->ee + jets with 70 < pT(Z) < 140 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Zee_PTV70_140_BFilter', # Z->ee + jets with 70 < pT(Z) < 140 GeV and a requirement for b-jets
                  #'Zee_PTV140_280_CVetoBVeto', # Z->ee + jets with 140 < pT(Z) < 280 GeV whilst vetoing c and b-jets
                  'Zee_PTV140_280_CFilterBVeto', # Z->ee + jets with 140 < pT(Z) < 280 GeV and a requirement for c-jets, whilst vetoing b-jets
                  'Zee_PTV140_280_BFilter', # Z->ee + jets with 140 < pT(Z) < 280 GeV and a requirement for b-jets
                  #'Zee_PTV280_500_CVetoBVeto', # Z->μμ + jets with 280 < pT(Z) < 500 GeV whilst vetoing c and b-jets
                  'Zee_PTV280_500_CFilterBVeto', # Z->ee + jets with 280 < pT(Z) < 500 GeV and a requirement for c-jets, whilst vetoing b-jets
                  'Zee_PTV280_500_BFilter', # Z->ee + jets with 280 < pT(Z) < 500 GeV and a requirement for b-jets 
                  'Zee_PTV500_1000', # Z->ee + jets with 500 < pT(Z) < 1000 GeV
                  #'Zee_PTV1000_E_CMS', # Z->ee + jets with 1000 GeV < pT(Z) < centre-of-mass energy
                  ]
    },                                                                                                                                                                                                                                                                                                                                                                                  

    'Other' : { # background with at least 2 prompt leptons, other than Z+jets and ttbar                                                                                                                                                                                
        'list' : [
                  'ZqqZll', # Z(->qq)Z(->ll)
                  #'single_top_wtchan', # Wt
                  #'single_antitop_wtchan', # Wt
                  #'lllv', # W(->lv)Z(->ll)
                  #'ttW',
                  #'llll', # ZZ->llll
                  #'llvv', # ZZ->llvv and WW->lvlv
                  #'Ztautau_PTV0_70_CVetoBVeto', # Z->ττ + jets with 0 < pT(Z) < 70 GeV whilst vetoing c and b-jets
                  #'Ztautau_PTV0_70_CFilterBVeto', # Z->ττ + jets with 0 < pT(Z) < 70 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Ztautau_PTV0_70_BFilter', # Z->ττ + jets with 0 < pT(Z) < 70 GeV and a requirement for b-jets
                  #'Ztautau_PTV70_140_CVetoBVeto', # Z->μμ + jets with 70 < pT(Z) < 140 GeV whilst vetoing c and b-jets
                  #'Ztautau_PTV70_140_CFilterBVeto', # Z->ττ + jets with 70 < pT(Z) < 140 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Ztautau_PTV70_140_BFilter', # Z->ττ + jets with 70 < pT(Z) < 140 GeV and a requirement for b-jets
                  #'Ztautau_PTV140_280_CVetoBVeto', # Z->μμ + jets with 140 < pT(Z) < 280 GeV whilst vetoing c and b-jets
                  #'Ztautau_PTV140_280_CFilterBVeto', # Z->ττ + jets with 140 < pT(Z) < 280 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Ztautau_PTV140_280_BFilter', # Z->ττ + jets with 140 < pT(Z) < 280 GeV and a requirement for b-jets
                  #'Ztautau_PTV280_500_CVetoBVeto', # Z->μμ + jets with 280 < pT(Z) < 500 GeV whilst vetoing c and b-jets
                  #'Ztautau_PTV280_500_CFilterBVeto', # Z->ττ + jets with 280 < pT(Z) < 500 GeV and a requirement for c-jets, whilst vetoing b-jets
                  #'Ztautau_PTV280_500_BFilter', # Z->ττ + jets with 280 < pT(Z) < 500 GeV and a requirement for b-jets
                  #'Ztautau_PTV500_1000', # Z->ττ + jets with 500 < pT(Z) < 1000 GeV
                  #'Ztautau_PTV1000_E_CMS' # Z->ττ + jets with 1000 GeV < pT(Z) < centre-of-mass energy
                 ]                                                                                                                                                                   
    },                                                                                                                                                                                           
    
}
