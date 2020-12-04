maps=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
wm_km_sc_aff=[2.72,11.61,11.74,4.42,4.47,7.55,3.81,3.44,4.29,40.91,6.71,3.13,3.55,7.16,6.98,7.99,2.40,2.48,2.37,2.68]
wm_px_sc_aff=[66.5,94.6,89.8,49.0,38.7,74.8,37.2,33.5,43.5,63.6,40.0,46.1,41.7,64.1,60.7,52.8,27.5,35.5,27.7,29.0]

results_sorted = sorted(zip(maps,wm_km_sc_aff), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
mean_error = sum(wm_km_sc_aff)/len(wm_km_sc_aff)
print("other median km",median_error)
print("other mean km",mean_error)

results_sorted = sorted(zip(maps,wm_px_sc_aff), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
mean_error = sum(wm_px_sc_aff)/len(wm_px_sc_aff)
print("other affine median px",median_error)
print("other affine mean px",mean_error)

import matplotlib.pyplot as plt
sheet_names = ["529","12","3","4","5",
                "6","7","8","9","10-18",
                "11",#"79a",
                "168","173","259",
                "327","328","329","330","331",
                "332","333","352","353","354",
                "355","356","357","358","377",
                "378","379","381","382","383",
                "402","403","404","405","406",
                "407","408","428","429","430",
                "431","432","433","434","571",
                "572","581","602"]
#["529","12","3","4","5","6","7","8","9","10","11","79a","168","173","259","327","328","382","383"]
error_ours = [127.425356,  808.127598, 405.070386, 349.664682, 509.326544, 
                 1977.204952, 499.567129, 333.353408, 114.199929, 347.203024, 
                 164.871981, # 18179.592263, 
                 191.158490,  75.028294, 52.330538, 
                 102.093746,  121.107682,  2067.600979, 92.453029, 1543.557139, 
                 91.655742,   51.111890,   129.926327, 87.675772, 171.246463, 
                 61.944273,   77.858277,   58.707810, 275.575773, 107.694777, 
                 1601.318240, 84.639471,   126.455419, 87.476919, 119.161954, 
                 1690.922109, 1239.491436, 62.393875, 126.555694, 99.391196, 
                 77.121876,   100.562299,  140.500505, 89.768985, 69.173359, 
                 64.417055,   73.200442,   83.135307, 64.073825, 1424.003588, 
                 85.241935,   82.453438,   73.992250]
errers_homography = {"529":94.565485,"12":707.172848,"3":1415.640065,"4":653.076558,"5":1006.338390,"6":2420.441656,"7":464.467480,"8":832.661051,"9":151.933902,
"10-18":378,"11":383.828872,"79a":24948.190689,"168":199.346165,"173":61.019486,"259":55.052516,"327":101.565729,"328":152.072666,"329":1974.843301,"330":181.804898,
"331":1879.637310,"332":164.339525,"333":114.531532,"352":292.974791,"353":96.665152,"354":121.666220,"355":79.816886,"356":130.483743,"357":57.380114,"358":292.208575,
"377":99.572036,"378":71.533923,"379":87.841745,"380":50.308258,"381":150.648295,"382":59.609603,"383":196.396860,"402":2070.228011,"403":229.837931,"404":66.206888,
"405":133.270741,"406":108.430550,"407":79.191592,"408":216.673433,"428":131.314116,"429":1130,"430":69.618423,"431":47.400478,"432":75.761663,"433":68.065370,
"434":1120.302031,"571":1144.550734,"572":223.555789,"581":45.933109,"602":64.762433}
error_homog = [errers_homography[x] for x in sheet_names]
error_homog_px = [x/6.4 for x in error_homog]
error_ours_px = [x/6.4 for x in error_ours]
print("ours median px",sorted(error_ours_px)[len(sorted(error_ours_px))//2])
print("ours mean px", sum(error_ours_px)/len(error_ours_px))
weinman_sc_tps=[61.3,93.3,81.8,33.8,32.5,74.5,34.0,29.4,42.8,56.7,41.9,50.5,53.1,51.4,43.1,59.8,27.7,26.5,26.2,27.8]
plt.boxplot([error_sorted,weinman_sc_tps,error_homog_px,error_ours_px], vert=False, showmeans=True, medianprops={"color":"r"})
plt.axhline(0, xmax=0, c="g", label="mean")
plt.axhline(0, xmax=0, c="r", label="median")
plt.xlabel("error [px]")
plt.yticks([1,2,3,4],["Howe et al. (affine)","Howe et al. (TPS)","ours (homography)","ours (affine)"])
# plt.xticks(range(len(sheet_names_sorted)),sheet_names_sorted,rotation=-90)
plt.legend()
plt.show()