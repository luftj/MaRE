
file="Glossar.tsv"
outfile="glossar.tex"
sep="\t"

output = []

with open(file, encoding="utf-8") as fr:
    fr.readline() # skip header
    for line in fr:
        _,entry_type,eintrag,plural,alias,engl,desc = line.split(sep)

        if eintrag == "":
            continue
        
        desc = desc.strip()

        if desc == "":
            continue

        if desc[-1] == ".":
            desc = desc[:-1]

        if engl != "":
            desc = f"Engl. \emph{{{engl}}}. {desc}"

        if alias != "":
            desc = f"Auch \emph{{{alias}}}. {desc}"

        # if entry_type == "akro":
        #     pass
        # else:
        key = eintrag.lower()
        key = key.replace("ü","ue")
        key = key.replace("ä","ae")
        key = key.replace("ö","oe")
        key = key.replace("ß","ss")
        key = key.split(")")[-1]
        key = key.strip()
        key = key.replace(" ","_")
        print(eintrag,key)
        bibtex = f"""\\newglossaryentry{{{key}}} {{\n\t\tname={{{eintrag}}},\n\t\tdescription={{\n\t\t\t{desc}}}"""
        if plural != "":
            bibtex += f",\n\t\tplural={{{plural}}}"
        if eintrag[0] in ["Ä","Ö","Ü"]:
            bibtex += f",\n\t\tsort={{{key}}}"
        bibtex += """\n}\n"""
        output.append(bibtex)
# print(output)

with open(outfile, "w", encoding="utf-8") as fw:
    fw.write("\\usepackage[toc,nonumberlist]{glossaries}\n")
    fw.write("\\makeglossaries\n\n")
    fw.writelines(output)