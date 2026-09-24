import json
from rag import llm_result


with open('data/eval_set.json', 'r') as f:
    eval_set = json.load(f)


correct = 0
for item in eval_set:
    result = llm_result(item["question"])
    
    if item["answerable"]:
        if item["expected_doc_ids"][0] in result:
            correct += 1
            
    else:
        if "don't have a reliable source" in result:
            correct += 1
            
print(f"{correct}/{len(eval_set)} correct")