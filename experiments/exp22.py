from matplotlib import pyplot as plt
import numpy as np

# section A: localisation accuracy / % maps successfully georeferenced

heitzler = 8/20
howe = 20/20
burt_24k = 30/30
burt_50k = 42/54
burt_250k = 17/17
burt_total = (30+42+17)/(30+54+17) # 0.881
ours_kdr100 = 542/656
ours_dk50 = 24/25
ours_tudr200 = 32/41
ours_total = (542+24+32)/(656+25+41) # 598/722 = 0.828
ours_tgis = 53/55
bahgat_runfola_synth = 92.2/100
bahgat_runfola_real = 84.7/100

# scores = {
#     "Heitzler et al.": heitzler,
#     "Howe et al.": howe,
#     "Burt et al.": burt_total,
#     "Luft & Schiewe": ours_tgis,
#     "Bahgat & Runfola": bahgat_runfola_real,
#     "vorl.": ours_total
# }
scores = {
    "[6]": heitzler,
    "[5]": howe,
    "[4]": burt_total,
    "[3]": ours_tgis,
    "[2]": bahgat_runfola_real,
    "[1]": ours_total
}
plt.barh(
    list(scores.keys()),
    scores.values(),
    color=["tab:blue"]*5+["tab:orange"],
    )
plt.xticks(np.arange(0,1.01,0.1),[f"{round(y*100)} %" for y in np.arange(0,1.01,0.1)])
plt.xlabel("Erfolgsquote")

filetype="pdf"
dpi_text = 1/72
fig_width=420
fig_height=250
dpi = 600
params = {'backend': filetype,
            'axes.labelsize': 11,
            'font.size': 11,
            'legend.fontsize': 9,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            # 'figure.figsize': [7, 4.5],
            'text.usetex': True,
            'figure.figsize': [fig_width*dpi_text,fig_height*dpi_text]
            }
plt.rcParams.update(params)
plt.savefig("soa_sucess_compare."+filetype,dpi=dpi, bbox_inches = 'tight')
# plt.show()
# exit()

plt.close()
# section B: georeerencing errors comparison

maps=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t"]
weinman_km_sc_aff=[2.72,11.61,11.74,4.42,4.47,7.55,3.81,3.44,4.29,40.91,6.71,3.13,3.55,7.16,6.98,7.99,2.40,2.48,2.37,2.68]
weinman_px_sc_aff=[66.5,94.6,89.8,49.0,38.7,74.8,37.2,33.5,43.5,63.6,40.0,46.1,41.7,64.1,60.7,52.8,27.5,35.5,27.7,29.0]
weinman_px_sc_tps=[61.3,93.3,81.8,33.8,32.5,74.5,34.0,29.4,42.8,56.7,41.9,50.5,53.1,51.4,43.1,59.8,27.7,26.5,26.2,27.8]

error_luft_schiewe = [127.425356,  808.127598, 405.070386, 349.664682, 509.326544, 
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
error_luft_schiewe = [x/6.4 for x in error_luft_schiewe]

# to do: calculate ours as RMSE of pixel errors
from eval_scripts.eval_helpers import load_errors_csv

exp_dir = "E:/experiments/e8"
errors_kdr100 = load_errors_csv(f"{exp_dir}/eval_georef_result.csv").values()
exp_dir = "E:/experiments/e12a"
errors_dk50 = load_errors_csv(f"{exp_dir}/eval_georef_result.csv").values()
exp_dir = "E:/experiments/e12b_preu"
errors_tudr200 = load_errors_csv(f"{exp_dir}/eval_georef_result.csv").values()
errors_bahgar_runfola_synth = [0.33,0.66]*10+[2,4]*10+[10,15]*13+[40,60,80]*8+[110]*7

print("median kdr 100",list(sorted(errors_kdr100))[len(errors_kdr100)//2])
print("median dk50",list(sorted(errors_dk50))[len(errors_dk50)//2])
print("median tüdr200",list(sorted(errors_tudr200))[len(errors_tudr200)//2])
print("median total",sorted(list(errors_kdr100)+list(errors_dk50)+list(errors_tudr200))[len((list(errors_kdr100)+list(errors_dk50)+list(errors_tudr200)))//2])

# to do: get beter Burt et al errors?
all_errors = {
    "Burt et al. 1:24k": [0.74,2.25],
    "Burt et al. 1:50k": [1.09,3.03],
    "Burt et al. 1:250k": [2.72,5.46],
    # "weinman_km_sc_aff": [2.72,11.61,11.74,4.42,4.47,7.55,3.81,3.44,4.29,40.91,6.71,3.13,3.55,7.16,6.98,7.99,2.40,2.48,2.37,2.68],
    "Howe et al. (TPS)": [61.3,93.3,81.8,33.8,32.5,74.5,34.0,29.4,42.8,56.7,41.9,50.5,53.1,51.4,43.1,59.8,27.7,26.5,26.2,27.8],
    "Howe et al. (Affin)": [66.5,94.6,89.8,49.0,38.7,74.8,37.2,33.5,43.5,63.6,40.0,46.1,41.7,64.1,60.7,52.8,27.5,35.5,27.7,29.0],
    "Luft \& Schiewe": error_luft_schiewe,
    "Bahgat \& Runfola synth. 2000px": [(x/100)*2000 for x in errors_bahgar_runfola_synth],
    "vorl. TÜDR200": [x/(3000/208) for x in errors_tudr200],
    "vorl. DK50": [x/(1000/191) for x in errors_dk50],
    "vorl. KDR100": [x/(3000/526) for x in errors_kdr100],
}
all_errors = {
    "[10]": [0.74,2.25],
    "[9]": [1.09,3.03],
    "[8]": [2.72,5.46],
    # "weinman_km_sc_aff": [2.72,11.61,11.74,4.42,4.47,7.55,3.81,3.44,4.29,40.91,6.71,3.13,3.55,7.16,6.98,7.99,2.40,2.48,2.37,2.68],
    "[7]": [61.3,93.3,81.8,33.8,32.5,74.5,34.0,29.4,42.8,56.7,41.9,50.5,53.1,51.4,43.1,59.8,27.7,26.5,26.2,27.8],
    "[6]": [66.5,94.6,89.8,49.0,38.7,74.8,37.2,33.5,43.5,63.6,40.0,46.1,41.7,64.1,60.7,52.8,27.5,35.5,27.7,29.0],
    "[5]": error_luft_schiewe,
    "[4]": [(x/100)*2000 for x in errors_bahgar_runfola_synth],
    "[3]": [x/(3000/208) for x in errors_tudr200],
    "[2]": [x/(1000/191) for x in errors_dk50],
    "[1]": [x/(3000/526) for x in errors_kdr100],
}

plt.rcParams.update(params)

bp = plt.boxplot(all_errors.values(), vert=False, showmeans=True, medianprops={"color":"r"})
print([x.get_xdata() for x in bp["medians"]])
plt.yticks(range(1,len(all_errors)+1),all_errors.keys())
plt.xlim(0,400)
plt.xlabel("Registrierungsfehler [px]")
plt.scatter([],[], c="g", marker="^", label="Mittelwert")
plt.axhline(10, xmax=0, c="r", label="Median")
plt.legend(loc='lower right')
# plt.show()
plt.savefig("soa_registration_compare."+filetype,dpi=dpi, bbox_inches = 'tight')

exit()
results_sorted = sorted(zip(maps,weinman_km_sc_aff), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
mean_error = sum(weinman_km_sc_aff)/len(weinman_km_sc_aff)
print("other median km",median_error)
print("other mean km",mean_error)

results_sorted = sorted(zip(maps,weinman_px_sc_aff), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
mean_error = sum(weinman_px_sc_aff)/len(weinman_px_sc_aff)
print("other affine median px",median_error)
print("other affine mean px",mean_error)

import matplotlib.pyplot as plt


params = {'backend': 'pdf',
        'axes.labelsize': 8,
        'font.size': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.figsize': [7, 4.5],
        'text.usetex': False,
        # 'figure.figsize': [420.0*1.0/72.27,250.0*1.0/72.27  ]
        }
dpi = 600
plt.rcParams.update(params)
plt.figure(figsize=(18/2.54,3.5))

plt.boxplot([error_sorted,weinman_px_sc_tps,error_homog_px,error_ours_px], vert=False, showmeans=True, medianprops={"color":"r"})
plt.scatter([],[], c="g", marker="^", label="mean")
plt.axhline(0, xmax=0, c="r", label="median")
plt.xlabel("georeferencing RMSE [px]")
plt.yticks([1,2,3,4],["Howe et al. (affine)","Howe et al. (TPS)","ours (homography)","ours (affine)"])
# plt.xticks(range(len(sheet_names_sorted)),sheet_names_sorted,rotation=-90)
plt.legend()
plt.savefig('figures_tgis/figure_8.pdf',dpi=dpi, bbox_inches = 'tight')
plt.show()