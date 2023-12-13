# VH_AC_BDTClassifier_v3
The generator of BDT for seperating SM Higgs and AC Higgs in VHMET(H->gg) channel via xgboost 

Example of usage:
    ```python3 main.py -n "acbdt_fa31d0" -ac "fa3" -x 1 -op 1```

where:
-n: name of the BDT
-ac: name of the AC parameter, should be "fa3", "fa2" or "fL1"
-x: 1 for output the xml file to be used in TMVA, 0 for not outputting
-op: 1 for output the plots of the BDT, 0 for not outputting
