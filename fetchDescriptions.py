import re
import requests


def generate_url(model, layer, feature):
    base_url = "https://www.neuronpedia.org"
    return f"{base_url}/{model}/{layer}/{feature}"


def generate_url_sae(model,sae,feature):
    """_summary_

    Args:
        model (str): e.g 'gpt2-small'
        sae (str): e.g  '8-res-jb' where 8 is the layer
        feature (str/int): number (eg index) of the feature.

    Returns:
        str: url to neuronpedia, eg:  https://www.neuronpedia.org/gpt2-small/7-res-jb/9589
    """
    #8-res-jb  ... where 8 is the layer
    base_url = "https://www.neuronpedia.org"
    return f"{base_url}/{model}/{sae}/{feature}"


def request(url):
    resp = requests.get(url)
    return resp


def scrape_data(data):
    """Returns empty str or a description of a Neuron.
    """
    ## Hacky; search html page for text. 
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