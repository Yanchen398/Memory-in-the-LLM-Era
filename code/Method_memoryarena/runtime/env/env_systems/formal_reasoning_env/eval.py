import os

import json
import sys
import tiktoken
import numpy as np
import pdb
import logging


# each jsonl is a problem with multiple subqueries
def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

#process all jsonl files in a directory
def process_all_papers_for_method(path, logger=None):
    paper_list = os.listdir(path)
    data = {}
    max_len=0
    for paper in paper_list:
        paper_path = os.path.join(path, paper, "result.jsonl")
        if os.path.isfile(paper_path) and paper_path.endswith('.jsonl'):
            print(f"Process paper: {paper}")
            d = load_jsonl(paper_path)
            data[paper] = (d, len(d))
            max_len = max(max_len, len(d))
    logger.info(f"Max length of logs among papers: {max_len}")            
    """
    {
        "aaa.pdf": ({subquery1, subquery2, ...}, length),
        ...
    }
    """
    return data, max_len

def get_passrate_at_k(data, logger=None):
    tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
    paper_passrate={}
    paper_passrate_at_k=[0]*20
    paper_count_at_k=[0]*20
    min_length=100000 # all paper min length
    for k,v in data.items():
        logs, length = v
        correct_count = 0
        tottime=[]
        memorytokens=[]
        for subquery_id, log in enumerate(logs):
            if log['is_correct']:
                correct_count += 1
                paper_passrate_at_k[subquery_id] += 1
            paper_count_at_k[subquery_id] += 1
            tottime.append(log.get('time',0))
            memory_len=tokenizer.encode(log.get('memory_context',''), disallowed_special=())
            memorytokens.append(len(memory_len))
        min_length = min(min_length, length)
        if logs[-1].get('is_correct', True):
            is_paper_correct=True
        else:            
            is_paper_correct=False       
        progress_score = correct_count / length
        paper_passrate[k] = {"progress_score": progress_score, #progress rate at paper k
                            "avg_length": np.mean(memorytokens),
                            "max_length": np.max(memorytokens),
                            "avg_time": np.mean(tottime),
                            "data_time":np.sum(tottime),
                            "is_paper_correct": is_paper_correct,
                            }
        logger.info(f"Paper {k}: Is_Paper_Correct = {is_paper_correct}, Progress Rate = {progress_score:.2f} ({correct_count}/{length})")
    return paper_passrate, paper_passrate_at_k, paper_count_at_k, min_length

def print_result(paper_passrate, paper_passrate_at_k, paper_count_at_k, min_length, path, logger=None):
    ps=[]
    passrate=[]
    avgl=[]
    avgt=[]
    maxll=[]
    totll=[]
    datatime=[]
    for k, re in paper_passrate.items():
        
        ps.append(re['progress_score'])
        avgl.append(re['avg_length'])
        avgt.append(re['avg_time'])
        maxll.append(re['max_length'])
        # totll.append(re['tot_length'])
        datatime.append(re['data_time'])
        passrate.append(re['is_paper_correct'])

    prate=[]
    for k in range(len(paper_passrate_at_k)):
        if paper_count_at_k[k]>0:
            prate.append(paper_passrate_at_k[k]/paper_count_at_k[k])
    cummulative_prate=[]
    for k in range(len(paper_passrate_at_k)):
        if paper_count_at_k[k]>0:
            pr_sofar=np.sum(paper_passrate_at_k[:k+1])
            cnt_sofar=np.sum(paper_count_at_k[:k+1])
            cummulative_prate.append(pr_sofar/cnt_sofar)
        
    logger.info(f"Progress Score len={len(ps)}, Pass Rate@k len={len(prate)}, Or you can stop at min length k={min_length})")
    logger.info(f"Overall Average Passrate: {np.mean(passrate):.4f}, Avg Progress Score: {np.mean(ps):.2f}, Avg Memory Length: {np.mean(avgl):.2f}, Avg Time/session: {np.mean(avgt)}s, Avg Time/task: {np.mean(datatime)}s Max Memory Length: {np.max(maxll):.2f}")


    # save result:
    os.makedirs(path, exist_ok=True)
    saved={
            "overall_average_passrate": float(np.mean(passrate)),
            "avg_progress_score": float(np.mean(ps)),
            "average_session_time": float(np.mean(avgt)),
            "average_memory_length": float(np.mean(avgl)),
            "average_task_time": float(np.mean(datatime)),
            "memory_length": float(np.max(maxll)), # memory horizon = max context length in long-context agents
            "min_k": min_length,
            "passrate_at_k": [float(p) for p in prate],
            "cummulative_passrate_at_k": [float(p) for p in cummulative_prate],
            "passrate_at_min_k": [float(p) for p in prate[:min_length]],
            "cummulative_passrate_at_min_k": [float(p) for p in cummulative_prate[:min_length]]

        }
    with open(f"{path}/all_results.json", "w") as f:
        json.dump(saved, f, indent=4)
    

    print("=========result=============")
    for k, v in saved.items():
        print(f"{k}: {v}\n")
    print("============================")
    return np.mean(ps), np.mean(avgl), np.mean(avgt), np.max(maxll), prate, cummulative_prate

def eval_and_print_result(cfg, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)
    json_folder=cfg["output"]["json_output_dir"] 
    method=[f for f in os.listdir(json_folder) if os.path.isdir(os.path.join(json_folder, f))]
    logger.info(f"Results found in {json_folder}\n Num. of memory systems={len(method)}")
    for i in range(len(method)):
        logger.info(f"##Result for method: {method[i]}")
        data, maxlen=process_all_papers_for_method(os.path.join(json_folder, method[i]), logger=logger)
        paper_passrate, paper_passrate_at_k, paper_count_at_k, min_length=get_passrate_at_k(data, logger=logger)
        print_result(paper_passrate, paper_passrate_at_k, paper_count_at_k, min_length, os.path.join(json_folder, method[i]), logger=logger)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise NotImplementedError("Warning: No json config provided. Using default config.")
    cfg=json.load(open(sys.argv[1]))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    eval_and_print_result(cfg)
    
