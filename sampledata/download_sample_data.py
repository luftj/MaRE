import os
import urllib.request, urllib.parse

data_dir = "./sampledata/"
urls = "kdr100_wiki_links.txt"

with open(data_dir + urls) as fr:
    for line in fr:
        url = line.strip()
        print("downlaoding",url)
        filepath = data_dir + urllib.parse.unquote(url.split("/")[-1])
        if os.path.exists(filepath):
            print("file already exists")
            continue
        filepath,_ = urllib.request.urlretrieve(url,filepath)
        print(filepath)