import argparse
from matplotlib import pyplot as plt
import numpy as np

def plot_single_distribution(ranks,figure_path):
    print("  0: %f%%"  % (len([x for x in ranks if x == 0])/len(ranks)*100))
    print("< 5: %f%%"  % (len([x for x in ranks if x < 5])/len(ranks)*100))
    print("< 10: %f%%"  % (len([x for x in ranks if x < 10])/len(ranks)*100))
    perc_30 = len([x for x in ranks if x <= 30])/len(ranks)*100
    print("< 30: %f%%"  % (perc_30))
    perc_50 = len([x for x in ranks if x <= 50])/len(ranks)*100
    print("< 50: %f%%"  % (perc_50))
    print("< 100: %f%%" % (len([x for x in ranks if x < 100])/len(ranks)*100))
    perc_100 = len([x for x in ranks if x <= 100])/len(ranks)*100
    print("< mean: %f%%" % (len([x for x in ranks if x < sum(ranks)/len(ranks)])/len(ranks)*100))
    mean = sum(ranks)/len(ranks)
    print("mean rank: %f" % (mean))

    counts = np.bincount(ranks)
    counts = np.cumsum(counts)
    plt.plot(counts, label="Blätter unter den ersten r Hypothesen")
    # plt.vlines(mean,counts[0],counts[-1], color="red", label="r<%d (mean)"%mean)
    plt.vlines(30,counts[0],counts[-1], linestyles="--", color="red", 
                                            label="r<30 (%0.2f%%)"%perc_30)
    plt.vlines(50,counts[0],counts[-1], linestyles="-.", color="darkorange", 
                                            label="r<50 (%0.2f%%)"%perc_50)
    plt.vlines(100,counts[0],counts[-1], linestyles="dotted", color="gold", 
                                            label="r<100 (%0.2f%%)"%perc_100)

    # perc68 = np.searchsorted(counts,0.68*counts[-1])
    # plt.vlines(perc68,counts[0],counts[-1], linestyles="--", label="r<%d (68%%)"%perc68)
    # perc80 = np.searchsorted(counts,0.8*counts[-1])
    # plt.vlines(perc80,counts[0],counts[-1], linestyles="-.", label="r<%d (80%%)"%perc80)
    # perc90 = np.searchsorted(counts,0.9*counts[-1])
    # plt.vlines(perc90,counts[0],counts[-1], linestyles="dotted", label="r<%d (90%%)"%perc90)
    # perc99 = np.searchsorted(counts,0.99*counts[-1])
    # plt.vlines(perc99,counts[0],counts[-1], linestyles=(0,(1,10)), label="r<%d (99%%)"%perc99)

    plt.xlabel("Stelle (r) in der Ähnlichkeitsordnung")
    plt.ylabel("# Kartenblätter")
    # plt.title("rank distribution of %s %s" % (param_to_tune,val))
    plt.legend()
    plt.savefig(figure_path+"/index_ranks.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("result", help="path to index result file")
    args = parser.parse_args()

    ranks = []
    with open(args.result, encoding="utf-8") as fr:
        for line in fr:
            sheet, rank = line.strip().split(" : ")
            if rank != "-1":
                ranks.append(int(rank))

    plot_single_distribution(ranks,"docs/eval_diagrams")