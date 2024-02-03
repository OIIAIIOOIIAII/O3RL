import torch
import argparse
import replay_buffer
import json
import math
import os
from tqdm import tqdm
import gc

def load_data_to_replay(name,data_count,count):
    replay = replay_buffer.Replay((150,),(1,),
                                  max_size=data_count,
                                  has_next_action=True)


    for data in name:
        for i in range(len(data['observations'])-1):
            s = data['observations'][i]
            a = data['bandwidth_predictions'][i]

            if not math.isnan(data['video_quality'][i]) and not math.isnan(data['audio_quality'][i]):
                r = data['video_quality'][i] + data['audio_quality'][i]
            elif math.isnan(data['video_quality'][i]) and not math.isnan(data['audio_quality'][i]):
                r = data['audio_quality'][i]
            elif math.isnan(data['audio_quality'][i]) and not math.isnan(data['video_quality'][i]):
                r = data['video_quality'][i]
            else:
                r = 0
            term = False
            if i == len(data['observations']) - 2:
                term = True
            sp = data['observations'][i + 1]
            ap = data['bandwidth_predictions'][i + 1]
            transition = replay_buffer.Transition(s, a, r, sp, ap, done=term)
            replay.append(transition)
    print (replay.__len__())
    path = 'data/' + 'test' + '_' + str(count) + '.pt'
    torch.save(replay,path)
    print ('Successfully saved at:',path)

    del replay
    gc.collect()



def batch_d4rl_to_replay(name):
    # path = 'D:/ACMMMSys/pythonProject/dataset'
    path = 'E:/data'
    testbed_path = path + '/testbed_dataset'
    emulated_path = path + '/emulated_dataset'

    testbed_dataset = []
    emulated_dataset = []

    # bed_foldernames = os.listdir(testbed_path)
    bed_foldernames = ['testbed_dataset_chunk_2']
    emulated_foldernames = os.listdir(emulated_path)

    # print (bed_foldernames, emulated_foldernames)
    testbed_file_count = 500
    json_count = 0
    print (bed_foldernames)
    for folder in bed_foldernames:
        bed_name = os.listdir(testbed_path + '/' + folder)
        for name in tqdm(bed_name,desc='正在加载' + folder):
            try:
                with open(testbed_path + '/' + folder + '/' + name)as f:
                    if json_count == 16:
                        testbed_count = 0
                        for ele in testbed_dataset:
                            testbed_count += len(ele['observations'])
                        load_data_to_replay(testbed_dataset,testbed_count,testbed_file_count)
                        testbed_file_count += 1
                        testbed_dataset.clear()
                        json_count = 0
                    testbed_dataset.append(json.load(f))
                    json_count += 1
            except Exception:
                continue
        print(folder,'saved')

    print('testbed pt文件数量是:' + str(testbed_file_count))



    emulated_file_count = testbed_file_count
    json_count = 0
    for folder in emulated_foldernames:
        emulated_name = os.listdir(emulated_path + '/' + folder)
        for name in tqdm(emulated_name,desc='正在加载' + folder):
            try:

                with open(emulated_path + '/' + folder + '/' + name)as f:
                    if json_count == 16:
                        emulated_count = 0
                        for ele in emulated_dataset:
                            emulated_count += len(ele['observations'])
                        load_data_to_replay(emulated_dataset,emulated_count,emulated_file_count)
                        emulated_file_count += 1
                        emulated_dataset.clear()
                        json_count = 0
                    emulated_dataset.append(json.load(f))
                    json_count += 1
            except Exception:
                continue
        print(folder, 'saved')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='halfcheetah-random-v2')
    args = parser.parse_args()

    batch_d4rl_to_replay(args.name)