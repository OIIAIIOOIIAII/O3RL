import json
import os
from tqdm import tqdm
import math

path = 'E:/data'
testbed_path = path + '/testbed_dataset'
emulated_path = path + '/emulated_dataset'

action_data_dict = {
    'sum' : 0,
    'mean' : 0,
    'std' : 0,
    'max' : 0,
    'min' : 0,
    'count' : 0
}

bed_foldernames = os.listdir(testbed_path)
emulated_foldernames = os.listdir(emulated_path)

for folder in bed_foldernames:
    bed_name = os.listdir(testbed_path + '/' + folder)
    for name in tqdm(bed_name):
        try:
            with open(testbed_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['bandwidth_predictions']
                for ele in testbed_data:
                    action_data_dict['sum'] += ele
                    action_data_dict['count'] += 1
                    if ele > action_data_dict['max']:
                        action_data_dict['max'] = ele
                    if ele < action_data_dict['min']:
                        action_data_dict['min'] = ele

        except Exception:
            continue
    with open('action_guiyi_data.json', 'w') as f:
        json.dump(action_data_dict, f)
    print ('json saved')

for folder in emulated_foldernames:
    emulated_names = os.listdir(emulated_path + '/' + folder)
    for name in tqdm(emulated_names):
        try:
            with open(emulated_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['bandwidth_predictions']
                for ele in testbed_data:
                    action_data_dict['sum'] += ele
                    action_data_dict['count'] += 1
                    if ele > action_data_dict['max']:
                        action_data_dict['max'] = ele
                    if ele < action_data_dict['min']:
                        action_data_dict['min'] = ele

        except Exception:
            continue
    with open('action_guiyi_data.json', 'w') as f:
        json.dump(action_data_dict, f)
    print ('json saved')

action_data_dict['mean'] = action_data_dict['sum']/action_data_dict['count']

with open('action_mean_data.json', 'w') as f:
    json.dump(action_data_dict, f)
print('mean saved')

action_data_dict['sum'] = 0

for folder in bed_foldernames:
    bed_name = os.listdir(testbed_path + '/' + folder)
    for name in tqdm(bed_name):
        try:
            with open(testbed_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['bandwidth_predictions']
                for ele in testbed_data:
                    numbers = (ele - action_data_dict['mean']) ** 2
                    action_data_dict['sum'] += numbers
        except Exception:
            continue
    with open('action_guiyi_data.json', 'w') as f:
        json.dump(action_data_dict, f)
    print('json saved')

for folder in emulated_foldernames:
    emulated_names = os.listdir(emulated_path + '/' + folder)
    for name in tqdm(emulated_names):
        try:
            with open(emulated_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['bandwidth_predictions']
                for ele in testbed_data:
                    numbers = (ele - action_data_dict['mean']) ** 2
                    action_data_dict['sum'] += numbers
        except Exception:
            continue
    with open('action_guiyi_data.json', 'w') as f:
        json.dump(action_data_dict, f)
    print('json saved')

action_data_dict['std'] = math.sqrt(action_data_dict['sum']/action_data_dict['count'])

with open('action_guiyi_data.json','w') as f:
    json.dump(action_data_dict,f)