# Experiment 4: Robustheit gegen Occlusion/Verdeckung/fehlende Signaturen

from math import ceil
import os, shutil, glob

from eval_logs import eval_logs
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_retrieval_logs import make_summary
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

from experiments.exp3 import make_figure, get_all_results

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

# set config
shutil.move("config.py", "config.py.old")
shutil.copy("experiments/config_e4.py", "config.py")

occlusion_params = [300,500,700]
data_dir_base = "E:/data/osm_baseline_degraded/occlusion/"
out_dir_base = "E:/experiments/e4/"

try:
    # make degraded dataset
    for nv in occlusion_params:
        data_dir=f"{data_dir_base}{nv}/"
        try:
            # check for data
            os.makedirs(data_dir, exist_ok=False)
        except OSError:
            print(f"data for {nv} already present")
        else:
            cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir} --circles {nv}"
            os.system(cmd)

    for nv in occlusion_params:
        out_dir=f"{out_dir_base}{nv}/"
        if os.path.isfile(f"{out_dir}/index_result.csv"):
            print("index already present")
            continue
        images_list=f"{data_dir_base}{nv}/list.txt"
        os.makedirs(out_dir,exist_ok=True)

        # cmd = f"""python indexing.py --list {images_list} {sheets}"""
        # os.system(cmd)
        # shutil.move("index_result.csv", out_dir + "index_result.csv")
        from indexing import search_list
        lps = search_list(images_list)
        with open(f"{out_dir}/index_result.csv","w") as fp:
            for sheet,rank in lps:
                fp.write("%s : %d\n"%(sheet,rank))
        
    # determine necessary number of spatial verifications
    for nv in occlusion_params:
        images_list=f"{data_dir_base}{nv}/list.txt"
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir,exist_ok=True)

        restrict_hypos = 0
        with open(f"{out_dir}/index_result.csv") as fp:
            for line in fp:
                sheet,rank = line.split(" : ")
                rank=int(rank)
                if rank > restrict_hypos:
                    restrict_hypos = rank

        max_hypos = 30
        restrict_hypos = ceil(restrict_hypos/5)*5 # round to next higher step of 5
        restrict_hypos = max(restrict_hypos,5) # leave some room for errors :)
        restrict_hypos = min(restrict_hypos,max_hypos)
        print(f"occlusion amount {nv}: verifying {restrict_hypos} hypotheses")

        # restrict_hypos=20
        if len(os.listdir(f"{out_dir_base}/{nv}")) < 1000:
            # run georeferencing
            cmd = f""" python main.py {images_list} {sheets} -r {restrict_hypos}"""# --noimg
            os.system(cmd)
            for file in glob.glob(f"{out_dir_base}/tmp/*"):
                shutil.move(file,out_dir)
            os.rmdir(f"{out_dir_base}/tmp/")
        else:
            print("georeferencing has already been run for", nv)
            continue
    
    # run evaluation scripts
    for nv in occlusion_params:
        # evaluate retrieval hit rate
        out_dir=f"{out_dir_base}{nv}/"
        os.makedirs(out_dir, exist_ok=True)
        
        if os.path.isfile(f"{out_dir}/eval_result.csv"):
            print("already evaluated retrieval")
            continue
       
        eval_logs(out_dir)
        shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
    for nv in occlusion_params:
        # evaluate georef accuracy
        out_dir=f"{out_dir_base}/{nv}/"
        
        if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
            print("already evaluated georef")
            continue

        calc_and_dump(sheets, out_dir)

    # compare different runs
    results = []
    results.append(get_all_results("E:/experiments/e2/", 0)) # add baseline as first result
    for idx,nv in enumerate(occlusion_params):
        data_dir=f"{data_dir_base}{nv}/"
        
        with open(f"{data_dir}/occlusion.txt") as fp:
            vals = []
            for line in fp:
                sheet,occl_val = line.strip().split(",")
                vals.append(float(occl_val))
        percent_occlusion = sum(vals)/len(vals)
        
        out_dir=f"{out_dir_base}{nv}/"
        results.append(get_all_results(out_dir, percent_occlusion))
    
    # make comparison figure
    make_figure(results, out_dir_base, "Verdeckung", max_error=500)

finally:
    # reset config
    os.remove("config.py")
    shutil.move("config.py.old", "config.py")



# regular
# wrong preds []
# filtered 0 incorrect predictions
# 655 sheets analysed
# total mean error: 42.33741066896848 mae m
# median MAE: 10.075360889521775 mae m
# best sheets: [('3', 6.9042642027264485), ('14', 6.973964403602512), ('71', 7.33680765949793), ('201', 7.5145549127874425), ('142', 7.538780802223369)]
# worst sheets: [('40', 777.2643391713304), ('341', 796.6907091080516), ('35', 2212.366519945377), ('91', 4436.639362817567), ('15', 11363.994612954893)]
# sheets < 200m: 647
# sheets > 500m: 6
# sheets > 1000m: 3
# sheets > mean: 16

# 300:
# wrong preds ['15', '27']
# filtered 0 incorrect predictions
# 651 sheets analysed
# total mean error: 119.94018309137874 mae m
# median MAE: 13.24858882781352 mae m
# best sheets: [('303', 5.257616806650351), ('24', 5.43223339819265), ('217', 6.865281046181633), ('430', 6.878872433262637), ('9', 6.941538858310069)]
# worst sheets: [('79', 1798.8525471208318), ('43', 4477.501547750113), ('10', 4807.643921018021), ('25', 10976.250839544062), ('91', 23808.50522048239)]
# sheets < 200m: 625
# sheets > 500m: 26
# sheets > 1000m: 14
# sheets > mean: 26

# 500:
# wrong preds ['23', '27', '28', '54', '56', '57', '59', '91', '102', '109', '112', '116', '117', '121', '130', '142', '145', '172', '175', '187', '191', '207', '210', '213', '223', '253', '276', '291', '293', '319', '326', '356', '368', '390', '396', '404', '407', '410', '446', '456', '458', '475', '489', '492', '500', '504', '508', '539', '579', '597', '600', '606', '623', '630', '633']
# filtered 0 incorrect predictions
# 529 sheets analysed
# total mean error: 204.76709145494814 mae m
# median MAE: 16.785690429230506 mae m
# best sheets: [('418', 5.389394503208757), ('286', 6.408859406436027), ('629', 6.865897211313414), ('53', 7.027466249319804), ('156', 7.045680059267362)]
# worst sheets: [('10', 4617.161595605868), ('15', 4772.173224757404), ('35', 6201.887426955824), ('61', 9426.916027684925), ('292', 10063.861444520751)]
# sheets < 200m: 467
# sheets > 500m: 61
# sheets > 1000m: 25
# sheets > mean: 62
# 700:
# wrong preds ['3', '7', '9', '10', '21', '41', '43', '47', '59', '62', '75', '83', '91', '94', '98', '100', '104', '105', '112', '117', '121', '123', '125', '127', '128', '129', '131', '133', '136', '144', 
# '145', '153', '160', '161', '162', '163', '164', '165', '170', '182', '183', '184', '185', '189', '192', '197', '199', '200', '205', '206', '215', '216', '218', '220', '236', '244', '245', '249', '253', '257', '273', '274', '278', '280', '289', '292', '298', '300', '309', '311', '312', '325', '326', '329', '330', '337', '341', '342', '344', '346', '348', '350', '355', '359', '362', '367', '370', '373', '380', '384', '390', '396', '398', '403', '404', '410', '414', '418', '423', '424', '425', '432', '433', '436', '456', '457', '467', '469', '471', '480', '489', '500', '504', '505', '513', '514', '539', '546', '560', '564', '581', '585', '590', '597', '603', '605', '613', '616', '617', '619', '621', '632', '637', '638', '639', '641', '661', '662', '664', '667', '669', '674']
# filtered 0 incorrect predictions
# 310 sheets analysed
# total mean error: 407.4184887645741 mae m
# median MAE: 24.957295223119104 mae m
# best sheets: [('174', 4.990296213855567), ('51', 6.987751965036278), ('52', 7.513088007967332), ('451', 7.882663343337508), ('73', 8.000843355590936)]
# worst sheets: [('194', 2271.8043577310855), ('475', 3657.1306940146806), ('276', 5069.87187295342), ('20', 19443.532992265853), ('15', 20916.775099713705)]
# sheets < 200m: 233
# sheets > 500m: 74
# sheets > 1000m: 30
# sheets > mean: 75