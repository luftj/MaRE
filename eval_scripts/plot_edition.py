edition_list = "E:/data/deutsches_reich/wiki/highres/edition_list.txt"

sheet_names = ["529",#"12",
                "3","4","5",
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
                "572","581","602","380"]
#["529","12","3","4","5","6","7","8","9","10","11","79a","168","173","259","327","328","382","383"]
error_results = [127.425356, # 808.127598,
                 405.070386, 349.664682, 509.326544, 
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
                 85.241935,   82.453438,   73.992250, 58.626446]

editions = {}
with open(edition_list) as fr:
    for line in fr:
        line=line.strip()
        name,edition = line.split(", ")
        editions[name] = edition

print(editions)

total_mean_error = sum(error_results)/len(error_results)
print("total mean error: %f m" % total_mean_error)

results_sorted = sorted(zip(sheet_names,error_results), key=lambda tup: tup[1])
sheet_names_sorted = [x[0] for x in results_sorted]
error_sorted = [x[1] for x in results_sorted]

median_error = error_sorted[len(error_sorted)//2]
print("median error: %f m" % median_error)

eds = [editions[s] for s in sheet_names_sorted]
colours = [ "r" if e=="A" else "b" for e in eds]

import matplotlib.pyplot as plt

plt.bar(sheet_names_sorted, error_sorted,label="error, edition B",color=colours)
plt.bar([0], [0],label="error, edition A",color="r")
plt.axhline(total_mean_error, c="g", linestyle="--", label="mean")
plt.annotate("%.0f" % total_mean_error,(0,total_mean_error + 50))
plt.axhline(median_error, c="r", label="median")
plt.annotate("%.0f" % median_error,(0,median_error + 50))
plt.xticks(range(len(sheet_names_sorted)),sheet_names_sorted,rotation=-90)
plt.legend()
plt.xlabel("sheet")
plt.ylabel("error [m]")
plt.show()