import re
import requests


def generate_url(model, layer, feature):
    base_url = "https://www.neuronpedia.org"
    return f"{base_url}/{model}/{layer}/{feature}"

def request(url):
    resp = requests.get(url)
    return resp

def scrape_data(data):
    m = re.findall('explanations.*description(.*)authorId', data)

    if m:
        return " ".join(re.findall('(\w*)', m[0])).strip()
    return ""

if __name__ == "__main__":
    m = "gpt-small"
    l = 11
    f = 1623
    
    url = generate_url(m, l, f)
    data = request(url).text