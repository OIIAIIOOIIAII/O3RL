import glob
import json
import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data_dir = "./test_data"
    figs_dir = "./submission/test3"
    onnx_model = "./onnx/final15000.onnx"
    data_files = glob.glob(os.path.join(data_dir, f'*.json'), recursive=True)
    # data_files = os.listdir(data_dir)
    print (data_files)
    ort_session = ort.InferenceSession(onnx_model)

    for filename in tqdm(data_files, desc="Processing"):
        with open(filename, "r") as file:
            call_data = json.load(file)

        observations = np.asarray(call_data['observations'], dtype=np.float32)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)
        video_score = np.asarray(call_data['video_quality'])
        audio_score = np.asarray(call_data['audio_quality'])
        video_score[np.isnan(video_score)] = 0
        audio_score[np.isnan(audio_score)] = 0
        videoscore = video_score.mean()
        audioscore = audio_score.mean()

        baseline_model_predictions = []
        hidden_state, cell_state = np.zeros((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)
        for t in range(observations.shape[0]):
            feed_dict = {'obs': observations[t:t+1,:].reshape(1,1,-1),
                        'hidden_states': hidden_state,
                        'cell_states': cell_state
                        }
            bw_prediction, hidden_state, cell_state = ort_session.run(None, feed_dict)
            # print(bw_prediction)
            baseline_model_predictions.append(bw_prediction[0,0,0])
            # print(baseline_model_predictions)
        baseline_model_predictions = np.asarray(baseline_model_predictions, dtype=np.float32)
        fig = plt.figure(figsize=(8, 5))
        time_s = np.arange(0, observations.shape[0]*60,60)/1000
        # print(baseline_model_predictions)
        plt.plot(time_s, baseline_model_predictions/1000, label='Offline RL Baseline', color='g')
        plt.plot(time_s, bandwidth_predictions/1000, label='BW Estimator '+call_data['policy_id'], color='r')
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='k')
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Call Duration (second)")
        plt.title('video score:' + str(round(videoscore, 2)) + '   ' + 'audio score:' + str(round(audioscore, 2)))
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(figs_dir,os.path.basename(filename).replace(".json",".png")))
        plt.close()