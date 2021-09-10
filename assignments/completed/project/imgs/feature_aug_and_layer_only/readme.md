balanced data


extra features
rdf = rdf.Define("p", "sqrt(px*px+py*py+pz*pz)")
rdf = rdf.Define("pphi", "atan2(py, px)*TMath::RadToDeg()")

rdf = rdf.Define("r", "sqrt(vx*vy*vy+vz*vz)")
rdf = rdf.Define("phi", "atan2(vy, vx)*TMath::RadToDeg()")