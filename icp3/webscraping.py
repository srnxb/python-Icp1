from bs4 import BeautifulSoup
import urllib.request
import os

html = urllib.request.urlopen("https://en.wikipedia.org/wiki/List_of_state_and_union_territory_capitals_in_India")

soup = BeautifulSoup(html, "html.parser")
print(soup.title.string)
for link in soup.find_all('a'):
    print(link.get('href'))

# print(th)