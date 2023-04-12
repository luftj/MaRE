import os
from eval_scripts.eval_register import get_georef_error,compare_special_cases
from eval_scripts.eval_helpers import load_errors_csv

sheets = "E:/data/deutsches_reich/blattschnitt/blattschnitt_kdr100_fixed.geojson"
images_list = "E:/data/deutsches_reich/SLUB/cut/raw/list.txt"
annotations = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"
exp_dir = "E:/experiments/e8_oldseg/"
resultsfile = f"{exp_dir}/eval_georef_result.csv"
out_dir = "E:/experiments/e8b"
os.makedirs(out_dir, exist_ok=True)

errors = load_errors_csv(resultsfile)
translation_mapping = {
    "coast": "Küste",
    "nocoast": "Inland",
    "total": "Gesamt",
    "A": "Auflage A",
    "B": "Auflage B",
}
# look at special cases
with open(f"{out_dir}/special_cases_summary.txt","w") as fw:
    # coast vs no coast
    coast_annotation = "E:/data/deutsches_reich/SLUB/cut/coast.txt"
    compare_special_cases(errors, "coast", coast_annotation, out_dir, logfile=fw, translation=translation_mapping)

    print(file=fw)
    # edition
    editions_annotation = "E:/data/deutsches_reich/SLUB/cut/editions.txt"
    compare_special_cases(errors, "editions", editions_annotation, out_dir, logfile=fw, translation=translation_mapping)

    print(file=fw)
    # overedge
    editions_annotation = "E:/data/deutsches_reich/SLUB/cut/shapes.txt"
    compare_special_cases(errors, "shapes", editions_annotation, out_dir, logfile=fw)