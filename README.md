# VH_AC_BDTClassifier_v3
The generator of BDT for seperating SM Higgs and AC Higgs in VHMET(H->gg) channel via xgboost 
Please check the `setup/requirements.txt` for the required packages.

Example of usage:
    python3 main.py -n "acbdt_fa31d0" -ac "fa3" --xmlfile --outplot --gpu

where:
-n: name of the BDT  
-ac: name of the AC parameter, should be "fa3", "fa2" or "fL1"   
--xmlfile: save the BDT in xml format  
--outplot: save the output plot  
--gpu: use GPU to train the BDT  