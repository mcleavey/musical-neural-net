import requests
from bs4 import BeautifulSoup
import os
import urllib.request
import argparse



def get_pieces(http_file, composer):
    f=open(http_file)
    content=f.read()
    if content=="Replace with page source.":
        print("\n\n** Error **")
        print("Please replace http.txt with the page source of")
        print("https://www.classicalarchives.com/secure/downloads.html")
        print("or use --source to specify a different input file.")
        print("(First login and select the 100 pieces you want to collect.)\n\n")
        f.close()
        return
    f.close()
    
    results_page = BeautifulSoup(content,'lxml')
    all_span = results_page.find_all('span', attrs={'class': 'first-child'})
    
    hrefs=[]
    for span in all_span:
        a=span.find('a')
        if a:
            hrefs.append(a.get('href'))        
    hrefs = [h for h in hrefs if h[-3:]=="mid" ]

    directory="./composers/midi/"+composer
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    for link in hrefs:
        parts=link.split('/')
        filename=directory+"/"+parts[-1]
        local_filename, headers = urllib.request.urlretrieve(link, filename)
        print(local_filename, headers)
        
def main(source, composer):
    get_pieces(source, composer)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--composer", help="Composer name (output will go to ./composers/midi/<this composer>") 
    parser.add_argument("--source", help="Page source of https://www.classicalarchives.com/secure/downloads.html")
    args = parser.parse_args()
    if args.source:
        source="./http_source/"+args.source
    else:
        source="./http_source/http.txt"
        

    main(source, args.composer)
    
