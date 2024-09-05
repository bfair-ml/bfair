from bs4 import BeautifulSoup
from markdown import markdown


def md2text(input_md):
    html = markdown(input_md)
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    return text
