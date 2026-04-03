##### generate variants for position sensitivity analysis #####

import json
with open('./longmemeval_s.json', 'r') as f:
    data = json.load(f)

import copy
import random
data1 = copy.deepcopy(data)
data2 = copy.deepcopy(data)
data3 = copy.deepcopy(data)
data_new = [data1, data2, data3]
for m in range(3):
    for j in range(len(data)):
        sample = data[j]
        sample_size = len(sample['haystack_session_ids'])
        key_size = len(sample['answer_session_ids'])
        batch_size = (sample_size-key_size) // 3
        batch = []
        batch.append(list(range(0, batch_size)))
        batch.append(list(range(batch_size, 2*batch_size)))
        batch.append(list(range(2*batch_size, sample_size)))

        key_pos=[] 
        for id in sample['answer_session_ids']:
            key_pos.append(sample['haystack_session_ids'].index(id))
        key_dates = [sample['haystack_dates'][i] for i in key_pos]
        key_ids = [sample['haystack_session_ids'][i] for i in key_pos]
        key_sessions = [sample['haystack_sessions'][i] for i in key_pos]

        new_dates = [sample['haystack_dates'][i] for i in range(sample_size) if i not in key_pos]
        new_ids = [sample['haystack_session_ids'][i] for i in range(sample_size) if i not in key_pos]
        new_sessions = [sample['haystack_sessions'][i] for i in range(sample_size) if i not in key_pos]

        new_pos = sorted(random.sample(batch[m], len(key_pos)), reverse=True)
        for i in range(len(new_pos)):
            new_dates.insert(new_pos[i], key_dates[i])
            new_ids.insert(new_pos[i], key_ids[i])
            new_sessions.insert(new_pos[i], key_sessions[i])
        data_new[m][j]['haystack_dates'] = new_dates
        data_new[m][j]['haystack_session_ids'] = new_ids
        data_new[m][j]['haystack_sessions'] = new_sessions
data1 = data_new[0]
data2 = data_new[1]
data3 = data_new[2]

with open('./longmemeval_s_l1.json', 'w') as f:
    json.dump(data1, f, indent=4)
with open('./longmemeval_s_l2.json', 'w') as f:
    json.dump(data2, f, indent=4)
with open('./longmemeval_s_l3.json', 'w') as f:
    json.dump(data3, f, indent=4)


##### generate variants for 50% context scalability analysis #####

data_50 = copy.deepcopy(data)
for j in range(len(data)):
    sample = data[j]
    sample_len = len(sample['haystack_session_ids'])
    new_dates = []
    new_ids = []
    new_sessions = []
    count = 0
    for i in range(sample_len):
        if sample['haystack_session_ids'][i] in sample['answer_session_ids'] or i%2 == 0:
            new_dates.append(sample['haystack_dates'][i])
            new_ids.append(sample['haystack_session_ids'][i])
            new_sessions.append(sample['haystack_sessions'][i])
            count += 1
    print(count)
    data_50[j]['haystack_dates'] = new_dates
    data_50[j]['haystack_session_ids'] = new_ids
    data_50[j]['haystack_sessions'] = new_sessions

with open('./longmemeval_s_50.json', 'w') as f:
    json.dump(data_50, f, indent=4)

##### generate variants for 150% and 200% context scalability analysis #####
with open('./5_filler_sess/data_5_filler_sess.json', 'r') as f:
    sess_pool = json.load(f)
pool_len = len(sess_pool)

data150 = copy.deepcopy(data)
for j in range(len(data)):
    init_len = len(data[j]['haystack_session_ids'])
    add_len = init_len // 2
    new_sess_id = random.sample(range(pool_len), add_len)
    for i in range(add_len):
        new_sess = sess_pool[new_sess_id[i]]
        if new_sess['session_id'] not in data150[j]['haystack_session_ids']:
            data150[j]['haystack_session_ids'].append(new_sess['session_id'])
            data150[j]['haystack_sessions'].append(new_sess['session'])
            data150[j]['haystack_dates'].append(random.choice(data[j]['haystack_dates']))
with open('./longmemeval_s_150.json', 'w') as f:
    json.dump(data150, f, indent=4)

data200 = copy.deepcopy(data)
for j in range(len(data)):
    init_len = len(data[j]['haystack_session_ids'])
    add_len = init_len
    new_sess_id = random.sample(range(pool_len), add_len)
    for i in range(add_len):
        new_sess = sess_pool[new_sess_id[i]]
        if new_sess['session_id'] not in data200[j]['haystack_session_ids']:
            data200[j]['haystack_session_ids'].append(new_sess['session_id'])
            data200[j]['haystack_sessions'].append(new_sess['session'])
            data200[j]['haystack_dates'].append(random.choice(data[j]['haystack_dates']))
with open('./longmemeval_s_200.json', 'w') as f:
    json.dump(data200, f, indent=4)