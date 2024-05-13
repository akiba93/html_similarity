import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup as bs, Tag, NavigableString
from sentence_transformers import SentenceTransformer, util

def get_html(url1,url2):
    # initialize a session & set User-Agent as a regular browser
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36"

    # get the HTML content
    html1 = session.get(url1).content
    html2 = session.get(url2).content

    # parse HTML using beautiful soup
    soup1 = bs(html1, "html.parser")
    soup2 = bs(html2, "html.parser")
    
    return soup1,soup2

def html_similarity(soup1,soup2,weight):
    #head similarity
    head_simi = 0
    
    html1_head_descendants_list = [i for i in soup1.head.descendants]
    html2_head_descendants_list = [i for i in soup2.head.descendants]
    head_min_len = min(len(html1_head_descendants_list),len(html2_head_descendants_list))
    head_max_len = max(len(html1_head_descendants_list),len(html2_head_descendants_list))
    
    for i in range(head_min_len):
        tag1 = html1_head_descendants_list[i]
        tag2 = html2_head_descendants_list[i]
        if isinstance(tag1, Tag) and isinstance(tag2, Tag):
            head_simi += (0.33*name_similarity(tag1,tag2) + 0.33*attrs_similarity(tag1,tag2,0.5) 
                          + 0.33 * string_similarity(tag1,tag2))
        elif isinstance(tag1, NavigableString) and isinstance(tag2, NavigableString):
            head_simi += string_similarity(tag1,tag2)
        else:
            pass
    
    head_simi = head_simi / head_max_len
    
    #body similarity
    body_simi = 0
    
    html1_body_descendants_list = [i for i in soup1.body.descendants]
    html2_body_descendants_list = [i for i in soup2.body.descendants]
    body_min_len = min(len(html1_body_descendants_list),len(html2_body_descendants_list))
    body_max_len = max(len(html1_body_descendants_list),len(html2_body_descendants_list))
    
    for i in range(body_min_len):
        tag1 = html1_body_descendants_list[i]
        tag2 = html2_body_descendants_list[i]
        if isinstance(tag1, Tag) and isinstance(tag2, Tag):
            body_simi += (0.33*name_similarity(tag1,tag2) + 0.33*attrs_similarity(tag1,tag2,0.5) 
                          + 0.33 * string_similarity(tag1,tag2))
        elif isinstance(tag1, NavigableString) and isinstance(tag2, NavigableString):
            body_simi += string_similarity(tag1,tag2)
        else:
            pass
    
    body_simi = body_simi / body_max_len
    
    #html_similarity
    html_similarity = weight * head_simi + (1 - weight) * body_simi
    
    return html_similarity

#name
def name_similarity(tag1,tag2):
    if tag1.name == tag2.name:
        return 1
    else:
        return 0

#attrs
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(set(list1)) + len(set(list2))) - intersection
    return float(intersection) / union

def attrs_similarity(tag1,tag2,k):
    attrs1 = tag1.attrs
    attrs2 = tag2.attrs
    
    if attrs1 == {} and attrs2 == {}:
        return 1
    elif attrs1 == {} and attrs2 != {}:
        return 0
    elif attrs1 != {} and attrs2 == {}:
        return 0
    else:
        #tag1 attrs num:
        attrs1_key_set = set(attrs1.keys())
        attrs1_num_of_key = len(attrs1_key_set)
    
        #tag2 attrs num:
        attrs2_key_set = set(attrs2.keys())
        attrs2_num_of_key = len(attrs2_key_set)
    
        #shared key
        shared_key = attrs1_key_set & attrs2_key_set
        num_of_shared_key = len(shared_key)
    
        #key similarity -- jaccard distance
        key_similarity = num_of_shared_key/(attrs1_num_of_key + attrs2_num_of_key - num_of_shared_key)
    
        #value similarity
        value_similarity = 0
        for key in shared_key:
            attrs1_value = attrs1[key]
            attrs2_value = attrs2[key]
        
            if isinstance(attrs1_value, str) and isinstance(attrs2_value, str):
                value_similarity += sentence_transformers_similarity(attrs1_value,attrs2_value)
            elif isinstance(attrs1_value, list) and isinstance(attrs2_value, list):
                value_similarity += jaccard_similarity(attrs1_value,attrs2_value)
            elif type(attrs1_value) != type(attrs2_value):
                value_similarity += 0
        value_similarity = value_similarity/num_of_shared_key
    
        return k*key_similarity + (1-k)*value_similarity
    
#string
# use sentence similarity 
#https://www.sbert.net/docs/quickstart.html

def sentence_transformers_similarity(string1,string2):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Sentences are encoded by calling model.encode()
    emb1 = model.encode(string1)
    emb2 = model.encode(string2)
    cos_sim = util.cos_sim(emb1, emb2)
    return cos_sim.item()

def string_similarity(tag1,tag2):
    string1 = tag1.string
    string2 = tag2.string
    if string1 == None and string2 == None:
        return 1
    elif string1 == None and string2 != None:
        return 0
    elif string1 != None and string2 == None:
        return 0
    elif isinstance(string1, str) and isinstance(string2, str):
        return sentence_transformers_similarity(string1,string2)
    
if __name__ == "__main__":
    url1 = input("please enter the first url:")
    url2 = input("please enter the second url:")
    soup1,soup2 = get_html(url1,url2)
    simi = html_similarity(soup1,soup2,0.5)
    print("Two page similarity is %s"%simi)