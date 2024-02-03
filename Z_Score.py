
import json
import os
from tqdm import tqdm
import math

path = 'E:/data'
testbed_path = path + '/testbed_dataset'
emulated_path = path + '/emulated_dataset'

data_dict = {
    'sum' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'mean' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'std' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'max': [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'min' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    'count' : [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
}



bed_foldernames = os.listdir(testbed_path)
emulated_foldernames = os.listdir(emulated_path)

for folder in bed_foldernames:
    bed_name = os.listdir(testbed_path + '/' + folder)
    for name in tqdm(bed_name):
        try:
            with open(testbed_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['observations']
                for ele in testbed_data:

                    data = [ele[i:i+5] for i in range (0, len(ele), 5)]
                    for i in range(30):
                        data_dict['sum'][i]  += sum(data[i])
                        data_dict['count'][i] += 5
                        if max(data[i]) > data_dict['max'][i]:
                            data_dict['max'][i] = max(data[i])
                        if min(data[i]) < data_dict['min'][i]:
                            data_dict['min'][i] = min(data[i])
                # data_dict['mean'] = data_dict['sum']/count
                # print (data_dict)
            # print(data_dict['sum'][0])
        except Exception:
            print('error occurred')
            continue
    with open('30guiyi_data.json', 'w') as f:
        json.dump(data_dict, f)
    print ('json saved')

# with open('./guiyi_data.json','r')as f:
#     data_dict = json.load(f)
# print(data_dict)
for folder in emulated_foldernames:
    emulated_name = os.listdir(emulated_path + '/' + folder)
    for name in tqdm(emulated_name):
        try:
            with open(emulated_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['observations']
                for ele in testbed_data:

                    data = [ele[i:i + 5] for i in range(0, len(ele), 5)]
                    for i in range(30):
                        data_dict['sum'][i] += sum(data[i])
                        data_dict['count'][i] += 5
                        if max(data[i]) > data_dict['max'][i]:
                            data_dict['max'][i] = max(data[i])
                        if min(data[i]) < data_dict['min'][i]:
                            data_dict['min'][i] = min(data[i])
                # data_dict['mean'] = data_dict['sum']/count
                # print (data_dict)
            # print(data_dict['sum'][0])
        except Exception:
            print ('error occurred')
            continue
    with open('guiyi_data.json', 'w') as f:
        json.dump(data_dict, f)
    print('json saved')
for i in range (30):
    data_dict['mean'][i] = data_dict['sum'][i]/data_dict['count'][i]

with open('30mean_data.json', 'w') as f:
    json.dump(data_dict, f)
print('mean saved')

for folder in bed_foldernames:
    bed_name = os.listdir(testbed_path + '/' + folder)
    # bed_name = ['testbed_dataset_chunk_0']
    for name in tqdm(bed_name):
        try:
            with open(testbed_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['observations']
                for ele in testbed_data:
                    data = [ele[i:i + 5] for i in range(0, len(ele), 5)]
                    for i in range(30):
                        numbers = [(num - data_dict['mean'][i]) ** 2 for num in data[i]]
                        data_dict['sum'][i] += sum(numbers)
        except Exception:
            print ('error occured')
            continue
    with open('30new_guiyi_data.json', 'w') as f:
        json.dump(data_dict, f)
    print('json saved')


for folder in emulated_foldernames:
    emulated_name = os.listdir(emulated_path + '/' + folder)
    for name in tqdm(emulated_name):
        try:
            with open(emulated_path + '/' + folder + '/' + name) as f:
                testbed_data = json.load(f)['observations']
                for ele in testbed_data:
                    data = [ele[i:i + 5] for i in range(0, len(ele), 5)]
                    for i in range(30):
                        numbers = [(num - data_dict['mean'][i]) ** 2 for num in data[i]]
                        data_dict['sum'][i] += sum(numbers)
        except Exception:
            print ('error occured')
            continue
    with open('30new_guiyi_data.json', 'w') as f:
        json.dump(data_dict, f)
    print('json saved')

for i in range (30):
    data_dict['std'][i] = math.sqrt(data_dict['sum'][i]/data_dict['count'][i])

with open('30new_guiyi_data.json','w') as f:
    json.dump(data_dict,f)





