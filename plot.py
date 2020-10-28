import matplotlib.pyplot as plt

sheet_names = ["529","12","3","4","5",
                "6","7","8","9","10-18",
                "11","79a",
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
error_results = [127.425356,  808.127598, 405.070386, 349.664682, 509.326544, 
                 1977.204952, 499.567129, 333.353408, 114.199929, 347.203024, 
                 164.871981,  18179.592263, 
                 191.158490,  75.028294, 52.330538, 
                 102.093746,  121.107682,  2067.600979, 92.453029, 1543.557139, 
                 91.655742,   51.111890,   129.926327, 87.675772, 171.246463, 
                 61.944273,   77.858277,   58.707810, 275.575773, 107.694777, 
                 1601.318240, 84.639471,   126.455419, 87.476919, 119.161954, 
                 1690.922109, 1239.491436, 62.393875, 126.555694, 99.391196, 
                 77.121876,   100.562299,  140.500505, 89.768985, 69.173359, 
                 64.417055,   73.200442,   83.135307, 64.073825, 1424.003588, 
                 85.241935,   82.453438,   73.992250]
#[125.389971,1365.229497,438.477274,340.533221,518.974640,2152.846284,564.651450,327.211482,149.989941,347.203024,168.155410,18291.506132,193.079256,78.057878,52.330538,1513.676719,121.107682,134.220033,119.161954]
percent_blue = {"382": 4.83, "383": 4.44, "408": 5.45, "434": 4.49, "259": 4.4, "12": 1.25, "403": 3.93, "404": 4.61, "430": 3.57, "379": 4.66, "354": 2.92, "380": 6.39, "1-2": 1.85, "3": 2.34, "4": 4.91, "5": 3.61, "6": 0.84, "7": 3.4, "8": 3.24, "9": 2,"10-18": 2.73, "11": 1.73, "168": 3.1, "173": 4.3, "327": 4.68, "328": 3.33, "329": 3.03, "330": 3.71, "331": 2.99, "332": 3.5, "333": 3.62, "352": 3.84, "353": 4.32, "355": 4.05, "356": 2.97, "357": 3.32, "358": 3.84, "377": 3.81, "378": 3.23, "381": 3.88, "402": 3.77, "405": 3.49, "406": 4.03, "407": 4.23, "428": 4.93, "429": 3.23, "431": 5.03, "432": 5.33, "433": 4.28, "571": 1.71, "572": 1.94, "581": 3.63, "602": 2.38, "79a": 0.49, "529": 1.1, "269": 5.88}
scores = {"382": 144, "383": 95, "408" :49, "434":85, "259":101, "12":9, "403":40, "404":140, "430":94, "379":100, "354":58, "380":184, "1-2":46, "3":17, "4":36, "5":29, "6":15, "7":27, "8":9, "9":48, "10-18":129, "11":19, "168":48, "173":47, "327" :77, "328" :76, "329" :62, "330" :72, "331" :41, "332" :28, "333" :71, "352" :69, "353" :85, "355" :117, "356" :72, "357" :71, "358" :41, "377" :57, "378" :53, "381" :175, "402" :40, "405" :147, "406" :138, "407" :158, "428" :36, "429" :30, "431" :186, "432" :117, "433" :94, "571" :20, "572" :28, "581" :83, "602" :53, "79a" :5, "529" :64, "269" :32}

total_mean_error = sum(error_results)/len(error_results)
print("total mean error: %f m" % total_mean_error)

results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
print("median error: %f m" % median_error)

print("num maps", len(error_results))
# print(plt.rcParams.keys())
params = {'backend': 'ps',
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'text.usetex': True,
        'figure.figsize': [420.0*1.0/72.27,250.0*1.0/72.27  ]}
# plt.rcParams.update(params)
# plt.axes([0.125,0.2,0.95-0.125,0.95])
plt.subplot(2, 2, 1)
plt.bar(sheet_names_sorted, error_sorted,label="error")
plt.axhline(total_mean_error, c="g", linestyle="--", label="mean")
plt.annotate("%.0f" % total_mean_error,(0,total_mean_error + 100))
plt.axhline(median_error, c="r", label="median")
plt.annotate("%.0f" % median_error,(0,median_error + 100))
plt.xticks(range(len(sheet_names_sorted)),sheet_names_sorted,rotation=-90)
plt.legend()
plt.title("average error per sheet [m]")
plt.xlabel("sheet")
plt.ylabel("error [m]")

plt.subplot(2, 2, 2)
plt.title('% blue hist')

x = [percent_blue[s] for s in sheet_names]
x_abovemean = [percent_blue[s] for idx,s in enumerate(sheet_names) if error_results[idx] > total_mean_error]
y = error_results

plt.hist(x,[0,1,2,3,4,5,6], alpha=0.5, color="g", label="total")
plt.hist(x_abovemean,[0,1,2,3,4,5,6], color="r",label="above mean")
plt.ylabel("#sheets")
plt.xlabel("% blue px")
plt.legend()

plt.subplot(2, 2, 3)
plt.title('error vs % blue')
plt.scatter(x,y,label="error")
plt.axhline(total_mean_error, c="g", linestyle="--", label="mean")
plt.axhline(median_error, c="r", label="median")
plt.xlabel("blue px [%]")
plt.ylabel("error [m]")
# plt.boxplot(error_sorted, vert=False, showmeans=True, medianprops={"color":"r"})
# plt.axhline(total_mean_error, xmax=0, c="g", label="mean")
# plt.axhline(median_error, xmax=0, c="r", label="median")
plt.legend()

plt.subplot(2, 2, 4)
plt.title('matching score hist')

x = [scores[s] for s in sheet_names]
x_abovemean = [scores[s] for idx,s in enumerate(sheet_names) if error_results[idx] > total_mean_error]
x_abovemedian = [scores[s] for idx,s in enumerate(sheet_names) if error_results[idx] > median_error]
y = error_results

plt.hist(x,range(0,max(x),10), alpha=0.5, color="g", label="total")
# plt.hist(x_abovemedian,range(0,max(x),10), alpha=0.5, color="b")
plt.hist(x_abovemean,range(0,max(x),10), alpha=1, color="r",label="above mean")
plt.ylabel("#sheets")
plt.xlabel("matching score")
plt.legend()

plt.tight_layout()
# plt.savefig('fig1.eps')
plt.show()
exit()