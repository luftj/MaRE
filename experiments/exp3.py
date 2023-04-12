# Experiment 3: Robustheit gegen Rauschen

from math import ceil
import os, shutil, glob
from eval_logs import eval_logs
from experiments.eval_baseline_georef import calc_and_dump
from experiments.summary_retrieval_logs import make_summary
from experiments.summary_baseline_georef import load_results,filter_results,summary_and_fig

def get_all_results(out_dir, amount, key="mae m"):
    with open(f"{out_dir}/retrieval_summary.txt","w") as outfile:
            percent_correct = make_summary(f"{out_dir}/eval_result.csv", outfile=outfile)

    # georef scores
    georef_scores = load_results(out_dir+"baseline_georef_scores.csv")
    georef_scores = filter_results(georef_scores, f"{out_dir}/eval_result.csv")
    mean_mae, median_mae, maes = summary_and_fig(georef_scores,out_dir, key=key)

    # index scores should come from index_result, in eval_result they are limited to restrict number
    ranks = []
    with open(f"{out_dir}/index_result.csv") as fp:
        for line in fp:
            sheet,rank = line.split(" : ")
            ranks.append(int(rank))
    mean_index = sum(ranks)/len(ranks)
    median_index= sorted(ranks)[len(ranks)//2]
    max_rank = max(ranks)
    return {
        "amount": amount, 
        "Erfolgsquote": percent_correct, 
        "Median Index": median_index,
        "Mittel Index": mean_index,
        "Max. Index": max_rank,
        "Mittel Genauigkeit": mean_mae, 
        "Median Genauigkeit": median_mae,
        "Fehler": maes
    }

def make_figure(results, out_dir, degrade_type, max_error=400, x_type="percent"):
    from matplotlib import pyplot as plt
    # print(results)
    if x_type == "percent":
        xs = [int(x["amount"]*100) for x in results]
    elif x_type == "string":
        xs =  [x["amount"] for x in results]
    ret_scores = [x["Erfolgsquote"]*100 for x in results]
    reg_mean_scores = [x["Mittel Genauigkeit"] for x in results]
    reg_median_scores = [x["Median Genauigkeit"] for x in results]
    reg_scores = [x["Fehler"] for x in results]

    plt.close()
    ax = plt.gca()
    if x_type == "percent":
        plt.xlabel(f"Anteil {degrade_type} [%]")
    elif x_type == "string":
        plt.xlabel(f"{degrade_type}")
    ax.set_ylim(top=105)
    ax2 = ax.twinx()
    ax.plot(ret_scores,label="Erfolgsquote",c="r",marker=".")
    ax.set_ylabel('Erfolgsquote [%]',color="r")
    # ax2.boxplot(reg_scores, showfliers=False, showmeans=True, meanline=True, positions=range(len(xs)))
    # ax2.axhline(40, xmax=0, c="g", label="mean")
    # ax2.axhline(40, xmax=0, c="tab:orange", label="median")
    ax2.violinplot(reg_scores, positions=range(len(xs)), showextrema=False)#, showmeans=True, showmedians=True )
    ax2.set_ylim(0,max_error)
    # ax2.set_yscale("log")
    ax2.set_yticks(range(0,max_error,100))
    ax2.plot(reg_mean_scores,label="Mittel [m]",marker="x",color="g")
    ax2.plot(reg_median_scores,label="Median [m]",marker="+",color="tab:orange")
    ax2.set_ylabel('Fehler [m]')
    plt.legend()
    if "Mittel Genauigkeit Pixel" in results[0]:
        reg_mean_scores_px = [x["Mittel Genauigkeit Pixel"] for x in results]
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', 1.115))
        ax3.plot(reg_mean_scores_px,marker="x",label="Mittel [px]", linestyle="dashed",color="g")
        ax3.set_ylabel('Fehler [px]')
    if "Median Genauigkeit Pixel" in results[0]:
        reg_median_scores_px = [x["Median Genauigkeit Pixel"] for x in results]
        ax3.plot(reg_median_scores_px,marker="+",label="Median [px]", linestyle="dashed",color="tab:orange")
    plt.legend()
    plt.xticks(ticks=range(len(xs)),labels=xs)
    out_dir = "docs/eval_diagrams/"
    plt.savefig(f"{out_dir}/comparison_{degrade_type}.png",bbox_inches='tight')

if __name__ == "__main__":
    sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
    images_list_base = "E:/data/deutsches_reich/osm_baseline/list.txt"

    # set config
    shutil.move("config.py", "config.py.old")
    shutil.copy("experiments/config_e3.py", "config.py")

    noise_values = [0.2,0.4,0.6]
    noise_values = [0.6,0.7,0.8,0.9]
    noise_values = [0.6,0.7,0.8,0.9]
    # noise_values = [0.7]
    data_dir_base = "E:/data/osm_baseline_degraded/"
    out_dir_base = "E:/experiments/e3/"

    try:
        # make degraded dataset
        for nv in noise_values:
            data_dir=f"{data_dir_base}{nv}/"
            try:
                # check for data
                os.makedirs(data_dir, exist_ok=False)
            except OSError:
                print(f"data for {nv} already present")
            else:
                cmd = f"make_osm_baseline.py {sheets} {images_list_base} {data_dir} --saltpepper {nv}"
                os.system(cmd)

        for nv in noise_values:
            images_list=f"{data_dir_base}{nv}/list.txt"
            out_dir=f"{out_dir_base}{nv}/"
            if os.path.isfile(f"{out_dir}/index_result.csv"):
                print("index already present")
                continue
            os.makedirs(out_dir,exist_ok=True)

            from indexing import search_list
            lps = search_list(images_list)
            with open(f"{out_dir}/index_result.csv","w") as fp:
                for sheet,rank in lps:
                    fp.write("%s : %d\n"%(sheet,rank))
            
        # determine necessary number of spatial verifications
        for nv in noise_values:
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
            print(f"noise amount {nv}: verifying {restrict_hypos} hypotheses")

            # restrict_hypos=20
            if len(os.listdir(f"{out_dir_base}/{nv}")) < 2000:
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
        for nv in noise_values:
            # evaluate retrieval hit rate
            out_dir=f"{out_dir_base}{nv}/"
            os.makedirs(out_dir, exist_ok=True)
            
            if os.path.isfile(f"{out_dir}/eval_result.csv"):
                print("already evaluated retrieval")
                continue
        
            eval_logs(out_dir)
            shutil.move("eval_result.csv", f"{out_dir}/eval_result.csv")
        for nv in noise_values:
            # evaluate georef accuracy
            out_dir=f"{out_dir_base}{nv}/"
            
            if os.path.isfile(f"{out_dir}/baseline_georef_scores.csv"):
                print("already evaluated georef")
                continue

            calc_and_dump(sheets, out_dir)

        # compare different runs
        results = []
        results.append(get_all_results("E:/experiments/e2/", 0)) # add baseline as first result
        for nv in noise_values:
            out_dir=f"{out_dir_base}{nv}/"
            results.append(get_all_results(out_dir, nv))
        
        # make comparison figure
        make_figure(results, out_dir_base, "Rauschen", x_type="percent", max_error=300)

    finally:
        # reset config
        os.remove("config.py")
        shutil.move("config.py.old", "config.py")

# Data:
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

# 60%
# wrong preds []
# filtered 0 incorrect predictions
# 656 sheets analysed
# total mean error: 34.80672990875271 mae m
# median MAE: 10.68736699354585 mae m
# best sheets: [('452', 5.033264205929728), ('376', 5.794079947429086), ('227', 6.053206370927871), ('166', 6.142875126135221), ('152', 6.304378456034623)]
# worst sheets: [('108', 978.6062201535615), ('15', 1054.7495899166195), ('91', 1378.6248541152565), ('61', 1441.8078568954452), ('10', 4749.3032104086615)]
# sheets < 200m: 642
# sheets > 500m: 12
# sheets > 1000m: 4
# sheets > mean: 25

# 70%
# wrong preds []
# filtered 0 incorrect predictions
# 656 sheets analysed
# total mean error: 34.418788365501854 mae m
# median MAE: 11.037390582051986 mae m
# best sheets: [('110', 6.139247156719583), ('96', 6.322933934815424), ('28', 6.420264188123491), ('397', 6.5073788658249345), ('74', 6.751356970689852)]
# worst sheets: [('332', 895.1571194113186), ('109', 991.7065675036717), ('15', 1018.160624307992), ('91', 1864.5833275693149), ('10', 4509.285322207445)]
# sheets < 200m: 642
# sheets > 500m: 11
# sheets > 1000m: 3
# sheets > mean: 25

# 80%:
# wrong preds []
# filtered 0 incorrect predictions
# 656 sheets analysed
# total mean error: 38.52954615645857 mae m
# median MAE: 11.232684315904596 mae m
# best sheets: [('354', 5.536827064171674), ('88', 6.59085508350992), ('76', 6.8492884005740375), ('101', 6.861828706440106), ('313', 6.974802675281441)]
# worst sheets: [('91', 973.7812727528565), ('15', 1010.6592895137072), ('108', 1036.488265668053), ('61', 1201.4508855782756), ('10', 4551.038298854623)]
# sheets < 200m: 639
# sheets > 500m: 15
# sheets > 1000m: 4
# sheets > mean: 26

# 90%:
# wrong preds []
# filtered 0 incorrect predictions
# 656 sheets analysed
# total mean error: 40.24610289070324 mae m
# median MAE: 11.693690501741028 mae m
# best sheets: [('161', 5.64412724010902), ('24', 6.184919451052858), ('554', 6.366507565008392), ('324', 6.523995869281526), ('288', 6.7547427321733275)]
# worst sheets: [('108', 897.6610280090542), ('109', 1028.363937489744), ('91', 1052.1839836762701), ('61', 2029.492636436873), ('10', 4552.589340980166)]
# sheets < 200m: 638
# sheets > 500m: 15
# sheets > 1000m: 4
# sheets > mean: 28