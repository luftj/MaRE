import os,glob,json

pixlab_annots = glob.glob("experiments/annotations/[0-9][0-9][0-9].json")
print(pixlab_annots)
sheetfile_ext = ".bmp"
outfile = "E:/data/deutsches_reich/SLUB/cut/raw/annotations.csv"

with open(outfile,"w") as fw:
    fw.write("#filename,region_shape_attributes\n")
    for file in pixlab_annots:
        sheetname = os.path.basename(file).split(".")[0]
        with open(file) as fr:
            print(sheetname)
            data = json.load(fr)
            for point in data[0]["content"]:
                fw.write(f'{sheetname}{sheetfile_ext},"{{""cx"":{int(point["x"])},""cy"":{int(point["y"])}}}"\n')
        
        
        
