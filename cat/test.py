
import CAT
import json
import torch
import numpy as np
import pandas as pd
import scipy.stats




seed = 0
np.random.seed(seed)
torch.manual_seed(seed)







dataset = 'moocradar'

config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 8,
    'num_dim': 1, 
    'device': 'cuda:0',
    'policy':'',
    'betas': (0.9, 0.999),
}

test_length = 10

strategy = CAT.strategy.MFIStrategy()
# strategy = CAT.strategy.MFIStrategy()

ckpt_path = 'cat/ckpt/irt.pt'


test_triplets = pd.read_csv(f'datasets/MOOCRadar/test_triples.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'datasets/MOOCRadar/concept_map.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
metadata = json.load(open(f'datasets/MOOCRadar/metadata.json', 'r'))
test_data = CAT.dataset.AdapTestDataset(test_triplets, concept_map,
                                        metadata['num_test_students'], 
                                        metadata['num_questions'], 
                                        metadata['num_concepts'])


import warnings
warnings.filterwarnings("ignore")

model = CAT.model.IRTModel(**config)
model.init_model(test_data)
model.adaptest_load(ckpt_path)

S_sel ={}
for sid in range(test_data.num_students): 
    key = sid
    S_sel[key] = []

selected_questions={}

for it in range(1, test_length + 1):
    print(f"Iteration {it}")
    thetas = []
    for t in range(0, test_data.num_students): 
        theta = model.get_theta(t)[0]
        ability_percentile = scipy.stats.norm.cdf(theta) * 100
        thetas.append(float(ability_percentile))  
    
    selected_questions = strategy.adaptest_select(model, test_data)
    
    for student, question in selected_questions.items():
        test_data.apply_selection(student, question)       
    model.adaptest_update(test_data) 
    results = model.evaluate(test_data)
    print(results)
