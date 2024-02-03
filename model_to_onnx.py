import os
import numpy as np
from tqdm import tqdm
import onnxruntime as ort
import torch
import torch.nn as nn
import hydra
import torchsummary as summary

import policy_network

model_path = 'submission/test3/train_halfcheetah-medium-v2_0_pi_15000.pt'
model_data = torch.load(model_path)

pi_model = policy_network.Resnet(150, 2)
pi_model.load_state_dict(model_data)


BS = 1
T = 1
obs_dim = 150
hidden_size = 1
# create dummy inputs
inputs = np.asarray(np.random.uniform(0, 1, size=(BS, T, obs_dim)), dtype=np.float32)
dummy_inputs = torch.as_tensor(inputs)

print (dummy_inputs.shape)
initial_hidden_state = torch.zeros((BS,hidden_size))
initial_cell_state = torch.zeros((BS,hidden_size))

dummy_outputs,final_hidden_state,final_cell_state = pi_model(dummy_inputs, initial_hidden_state, initial_cell_state)

print(dummy_outputs)



# save onnx model
onnx_path = './submission/final models/final.onnx'
os.makedirs(os.path.dirname(onnx_path),exist_ok=True)
pi_model.to('cpu')
pi_model.eval()
torch.onnx.export(
    pi_model,
    (dummy_inputs[0:1,0:1,:],initial_hidden_state,initial_cell_state),
    onnx_path,
    opset_version=11,
    input_names=['obs', 'hidden_states', 'cell_states'],
    output_names=['output', 'state_out', 'cell_out'],
    # dynamic_axes={
    #     'input':{0: 'batch_size' , 1: 'seq_len'},
    #     'output':{0:'batch_size',1:'seq_len'}
    # }

)

ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
onnx_hidden_state, onnx_cell_state = (
np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
# online interaction: step through the environment 1 time step at a time
with torch.no_grad():
    for i in tqdm(range(inputs.shape[1])):
        torch_estimate, torch_hidden_state, torch_cell_state = pi_model(dummy_inputs[0:1, i:i + 1, :],
                                                                            torch_hidden_state, torch_cell_state)
        feed_dict = {'obs': inputs[0:1, i:i + 1, :], 'hidden_states': onnx_hidden_state,
                     'cell_states': onnx_cell_state}
        onnx_estimate, onnx_hidden_state, onnx_cell_state = ort_session.run(None, feed_dict)
        assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=1e-6), 'Failed to match model outputs!'
        assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
        assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'

    assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
    assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
    print("Torch and Onnx models outputs have been verified successfully!")
# pi.load_state_dict(model_data)

# ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# # onnx_hidden_state, onnx_cell_state = (
# # np.zeros((1, hidden_size), dtype=np.float32), np.zeros((1, hidden_size), dtype=np.float32))
# # torch_hidden_state, torch_cell_state = (torch.as_tensor(onnx_hidden_state), torch.as_tensor(onnx_cell_state))
# # online interaction: step through the environment 1 time step at a time
# with torch.no_grad():
#     torch_estimate = pi_model(dummy_inputs[0:1, 0: 1, :],
#                                                                             )
#     feed_dict = {'obs': dummy_inputs[0:1, 0: 1, :]}
#     onnx_estimate = ort_session.run(None, feed_dict)
#     assert np.allclose(torch_estimate.numpy(), onnx_estimate, atol=1e-6), 'Failed to match model outputs!'
#         # assert np.allclose(torch_hidden_state, onnx_hidden_state, atol=1e-7), 'Failed to match hidden state1'
#         # assert np.allclose(torch_cell_state, onnx_cell_state, atol=1e-7), 'Failed to match cell state!'
#
#     # assert np.allclose(torch_hidden_state, final_hidden_state, atol=1e-7), 'Failed to match final hidden state!'
#     # assert np.allclose(torch_cell_state, final_cell_state, atol=1e-7), 'Failed to match final cell state!'
#     print("Torch and Onnx models outputs have been verified successfully!")