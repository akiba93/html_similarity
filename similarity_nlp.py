import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from bs4 import BeautifulSoup as bs, Tag, NavigableString

def get_sentence_embedding(text,tokenizer,model):
    #tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    #model = AutoModelForCausalLM.from_pretrained("alecsharpie/codegen_350m_html")

    t_input = tokenizer(text, return_tensors="pt")
    
    if len(t_input['input_ids'][0]) > 2048:
        # if sentence is longer than max length
        input_id_chunks = list(t_input['input_ids'][0].split(2048))
        mask_chunks = list(t_input['attention_mask'][0].split(2048))
        for i in range(len(input_id_chunks)):
            # get required padding length
            pad_len = 2048 - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
            # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([input_id_chunks[i], torch.Tensor([tokenizer.pad_token_id] * pad_len)])
                mask_chunks[i] = torch.cat([mask_chunks[i], torch.Tensor([0] * pad_len)])
    
        # new input    
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)

        input_dict = {
            'input_ids': input_ids.long(),
            'attention_mask': attention_mask.int()
        }
    else:
        input_dict = t_input

    with torch.no_grad():
        last_hidden_state = model(**input_dict, output_hidden_states=True).hidden_states[-1]
    
    text_embedding = torch.mean(last_hidden_state,dim = 1).reshape(-1)
    return text_embedding

def text_embedding_similarity(text1_embedding,text2_embedding):
    text1_embedding_len = len(text1_embedding)
    text2_embedding_len = len(text2_embedding)

    max_len = max(text1_embedding_len,text2_embedding_len)

    if text1_embedding_len < max_len:
        pad_len = max_len - text1_embedding_len
        text1_embedding = torch.cat([text1_embedding, torch.Tensor([0] * pad_len)])
    elif text2_embedding_len < max_len:
        pad_len = max_len - text2_embedding_len
        text2_embedding = torch.cat([text2_embedding, torch.Tensor([0] * pad_len)])
    else:
        pass
    
    embedding_simi = torch.cosine_similarity(text1_embedding,text2_embedding, dim = 0).item()
    
    return embedding_simi

def html_similarity(soup1,soup2,model,tokenizer,weight):
    from bs4 import BeautifulSoup as bs, Tag, NavigableString
    
    #head similarity
    head_simi = 0
    
    html1_head_text_list = [str(i) for i in soup1.head.find_all()]
    html2_head_text_list = [str(i) for i in soup2.head.find_all()]
    head_min_len = min(len(html1_head_text_list),len(html2_head_text_list))
    head_max_len = max(len(html1_head_text_list),len(html2_head_text_list))
    
    for i in range(head_min_len):
        text1 = html1_head_text_list[i]
        text2 = html2_head_text_list[i]
        
        text1_embedding = get_sentence_embedding(text1,tokenizer,model)
        text2_embedding = get_sentence_embedding(text2,tokenizer,model)
        
        simi = text_embedding_similarity(text1_embedding,text2_embedding)
        head_simi += simi
    
    head_simi = head_simi / head_min_len
    
    #body similarity
    body_simi = 0
    
    html1_body_text_list = [str(i) for i in soup1.body.find_all()]
    html2_body_text_list = [str(i) for i in soup2.body.find_all()]
    body_min_len = min(len(html1_body_text_list),len(html2_body_text_list))
    body_max_len = max(len(html1_body_text_list),len(html2_body_text_list))
    
    for i in range(body_min_len):
        text1 = html1_body_text_list[i]
        text2 = html2_body_text_list[i]
        
        text1_embedding = get_sentence_embedding(text1,tokenizer,model)
        text2_embedding = get_sentence_embedding(text2,tokenizer,model)
        
        simi = text_embedding_similarity(text1_embedding,text2_embedding)
        body_simi += simi
    
    body_simi = body_simi / body_max_len
    
    #html_similarity
    html_similarity = weight * head_simi + (1 - weight) * body_simi
    
    return round(html_similarity,5)

def get_html(url1,url2):
    import requests
    from bs4 import BeautifulSoup as bs, NavigableString
    from urllib.parse import urljoin

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

if __name__ == "__main__":
    url1 = input("please enter the first url:")
    url2 = input("please enter the second url:")
    soup1,soup2 = get_html(url1,url2)
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-multi")
    model = AutoModelForCausalLM.from_pretrained("alecsharpie/codegen_350m_html")
    simi = html_similarity(soup1,soup2,model,tokenizer,0.5)
    print("Two page similarity is %s"%simi)