import urllib.request, urllib.parse
urls = "kdr100_wiki_links.txt"

with open(urls) as fr:
    for line in fr:
        url = line.strip()
        print(url)
        filepath = urllib.parse.unquote(url.split("/")[-1])
        filepath,_ = urllib.request.urlretrieve(url,filepath)
        print(filepath)