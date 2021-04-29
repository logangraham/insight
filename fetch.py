"""
Uses the UKRI GTR API to search and store abstracts as a json file.

You should probably ignore this due to current limitations:
- Requires a search term
- Not implemented: multi-page search
- Doesn't filter duplicate grants or grants with no abstract.
"""

import requests
import json


def fetch_projects(term, n=100, **kwparams):
    """
    Return a list of projects and their awards.

    :param term: str
    :param n: int
    :param kwparams: dict
    :return: list
    """
    ## determine number of page requests to make
    times, remainder = divmod(n, 100)
    if remainder > 0:
        times += 1

    ## search projects
    url = "https://gtr.ukri.org/search/project"
    params = {"term": term,
              "fetchSize": 100,
              **kwparams}
    projects = requests.get(url,
                            params=params,
                            headers={"accept": "application/json"}).json()['searchResult']['results']
    projects = [p['projectComposition']['project'] for p in projects]
    projects = [p for p in projects if p['fund']['valuePounds'] > 0]
    return projects

def fetch_info(url):
    """
    Returns the title, abstract, and award size.

    :param pid: [string or int] project id
    :return: [dict] dict ot title, abstract, and award size
    """
    project = requests.get(url, headers={"accept": "application/json"}).json()
    project = project['projectOverview']['projectComposition']
    return project

def fetch_and_store(term, n, dir="data/", **kwparams):
    """
    Fetch all related grants and store them in a json file.

    :param term: str
    :param n: int
    :param kwparams: dict
    :param dir: str
    :return: dict
    """
    projects = fetch_projects(term, n, **kwparams)
    pdict = {}
    for p in projects:
        pdict[p['id']] = fetch_info(p['url'])
    filepath = dir + "projects.json" if dir is not None else "projects.json"
    with open(filepath, 'w') as f:
        json.dump(pdict, f)
    return pdict

def filter_and_store(filepath, dir="data/"):
    """
    Filters projects and stores abstract, title, and amount.

    :param filepath:
    :param target:
    :return:
    """
    with open(filepath, "r") as f:
        d = json.load(f)

    for project in d:
        if len(d[project]['project']['abstractText']) > 200:
            newd[project] = {}
            newd[project]['abstract'] = d[project]['project']['abstractText']
            newd[project]['title'] = d[project]['project']['title']
            newd[project]['amount'] = d[project]['project']['fund']['valuePounds']

    with open(dir + "filtered_projects.json", "w") as f:
        json.dump(newd, f)

def get_sentences(filepath="data/filtered_projects.json"):
    with open(filepath, "r") as f:
        data = json.load(f)
    abstracts = [data[k]['abstract'].replace("\n", " ").split(". ") for k in data]
    sentences = [(item + ".").strip() for sublist in abstracts for item in sublist]
    return sentences


if __name__ == "__main__":
    fetch_and_store("machine learning", 100, "./data/projects.json")