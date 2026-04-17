import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
import os
from torch.utils.data import TensorDataset
import torch.nn as nn
from transformers import EncodecModel, EncodecConfig




if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

def seed_everything(seed):
    """It sets all the seeds for reproducibility.

    Args:
    ----------
    seed : int
        Seed for all the methods
    """
    print("Setting seeds")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WaveformDataset(TensorDataset):
    def __init__(self, waveform, nameTrace, label, target):
        self.waveform = waveform
        self.nameTrace = nameTrace
        self.label = label
        self.target = target
    def __len__(self):
        return len(self.nameTrace)
    def __getitem__(self, idx):
        waveform = self.waveform[idx]
        nameTrace = self.nameTrace[idx]
        label = self.label[idx]
        target = self.target[idx]
        waveform_ret=waveform#.cpu().clone().detach()
        nameTrace_ret=nameTrace#.cpu().clone().detach()
        label_ret=label#.cpu().clone().detach()
        target_ret=target#.cpu().clone().detach()
        return waveform_ret, nameTrace_ret, label_ret, target_ret
    

def normalize_dataset(inputs):
    """
    Args:
    ----------
    inputs : 
            torch.tensor of shape (channels, length signal) or (length dataset, channels, length signal)
    
    """
    print("The normalization is event-based, working on the 3 channels")
    in_max, _= torch.max(torch.abs(inputs.reshape(inputs.shape[0],-1)), axis=1, keepdims=True)
    in_max[in_max == 0.0] = 1e-10
    in_norm = inputs.reshape(inputs.shape[0],-1) / in_max
    inputs_norm = in_norm.reshape(inputs.shape).to(torch.float32)
    return inputs_norm

def generate_label(data, phase_list, mask=None, label_shape="gaussian", label_width=100):
    target = np.zeros_like(data)

    if label_shape == "gaussian":
        label_window = np.exp(
            -((np.arange(-label_width // 2, label_width // 2 + 1)) ** 2)
            / (2 * (label_width / 5) ** 2)
        )
    elif label_shape == "triangle":
        label_window = 1 - np.abs(
            2 / label_width * (np.arange(-label_width // 2, label_width // 2 + 1))
        )
    else:
        print(f"Label shape {label_shape} should be guassian or triangle")
        raise

    for i, phases in enumerate(phase_list):
        for j, idx_list in enumerate(phases):
            for idx in idx_list:
                if np.isnan(idx):
                    continue
                idx = int(idx)
                if (idx - label_width // 2 >= 0) and (idx + label_width // 2 + 1 <= target.shape[0]):
                    target[idx - label_width // 2 : idx + label_width // 2 + 1, j, i + 1] = label_window

        target[..., 0] = 1 - np.sum(target[..., 1:], axis=-1)
        if mask is not None:
            target[:, mask == 0, :] = 0

    return target



def random_shift(sample, itp, its, itp_old=None, its_old=None, shift_range=None, sampling_rate = 100):
    # anchor = np.round(1/2 * (min(itp[~np.isnan(itp.astype(float))]) + min(its[~np.isnan(its.astype(float))]))).astype(int)
    min_event_gap= 3 * sampling_rate
    flattern = lambda x: np.array([i for trace in x for i in trace], dtype=float)
    shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
    itp_flat = flattern(itp)
    its_flat = flattern(its)
    if (itp_old is None) and (its_old is None):
        hi = np.round(np.median(itp_flat[~np.isnan(itp_flat)])).astype(int)
        lo = -(sample.shape[0] - np.round(np.median(its_flat[~np.isnan(its_flat)])).astype(int))
        if shift_range is None:
            shift = np.random.randint(low=lo, high=hi + 1)
        else:
            shift = np.random.randint(low=max(lo, shift_range[0]), high=min(hi + 1, shift_range[1]))
    else:
        itp_old_flat = flattern(itp_old)
        its_old_flat = flattern(its_old)
        itp_ref = np.round(np.min(itp_flat[~np.isnan(itp_flat)])).astype(int)
        its_ref = np.round(np.max(its_flat[~np.isnan(its_flat)])).astype(int)
        itp_old_ref = np.round(np.min(itp_old_flat[~np.isnan(itp_old_flat)])).astype(int)
        its_old_ref = np.round(np.max(its_old_flat[~np.isnan(its_old_flat)])).astype(int)
        # min_event_gap = np.round(min_event_gap*(its_ref-itp_ref)).astype(int)
        # min_event_gap_old = np.round(min_event_gap*(its_old_ref-itp_old_ref)).astype(int)
        if shift_range is None:
            hi = list(range(max(its_ref - itp_old_ref + min_event_gap, 0), itp_ref))
            lo = list(range(-(sample.shape[0] - its_ref), -(max(its_old_ref - itp_ref + min_event_gap, 0))))
        else:
            lo_ = max(-(sample.shape[0] - its_ref), shift_range[0])
            hi_ = min(itp_ref, shift_range[1])
            hi = list(range(max(its_ref - itp_old_ref + min_event_gap, 0), hi_))
            lo = list(range(lo_, -(max(its_old_ref - itp_ref + min_event_gap, 0))))
        if len(hi + lo) > 0:
            shift = np.random.choice(hi + lo)
        else:
            shift = 0

    shifted_sample = np.zeros_like(sample)
    if shift > 0:
        shifted_sample[:-shift, ...] = sample[shift:, ...]
    elif shift < 0:
        shifted_sample[-shift:, ...] = sample[:shift, ...]
    else:
        shifted_sample[...] = sample[...]

    return shifted_sample, shift_pick(itp, shift), shift_pick(its, shift), shift

def cut_window(sample, target, itp, its, select_range):
    shift_pick = lambda x, shift: [[i - shift for i in trace] for trace in x]
    sample = sample[select_range[0] : select_range[1]]
    target = target[select_range[0] : select_range[1]]
    return (sample, target, shift_pick(itp, select_range[0]), shift_pick(its, select_range[0]))


def calc_metrics(nTP, nP, nT):
    """
    Calculate precision, recall, and F1 score.

    Args:
        nTP: True positives.
        nP: Number of positive picks (predicted). (true positives + false positives)
        nT: Number of true picks (ground truth). (true positives + false negatives)

    Returns:
        A list [precision, recall, f1 score].
    """
    precision = nTP / nP if nP > 0 else 0
    recall = nTP / nT if nT > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return [precision, recall, f1]


def calc_performance(preds, labels, tol=10.0, dt=1.0, print_metrics=True):
    """
    Calculate the performance of predictions vs true labels.

    Args:
        preds: Tensor of predicted picks with shape (N, 2), where N is the number of samples and 2 refers to P and S phases.
        labels: Tensor of true picks with shape (N, 2), where N is the number of samples and 2 refers to P and S phases.
        tol: Tolerance in samples (default is 10, corresponding to 0.1 seconds for a 100 Hz sampling rate), according to the paper.
        dt: Time scaling factor (default is 1.0).

    Returns:
        metrics: Dictionary with precision, recall, F1 score for P and S phases.
        diff_p: Residuals for P-phase picks.
        diff_s: Residuals for S-phase picks.
    """
    assert preds.shape == labels.shape, "Predictions and labels must have the same shape."

    metrics = {}
    phase_names = ["P", "S"]
    diff_p = []  # To store residuals for P-phase
    diff_s = []  # To store residuals for S-phase
    residual = []

    for phase_idx, phase_name in enumerate(phase_names):
        true_positive, positive, true = 0, 0, 0
        

        # Loop over all samples
        for i in range(preds.shape[0]):
            true_value = labels[i, phase_idx]
            pred_value = preds[i, phase_idx]

            # Ignore picks where the label is zero (assumed to be no pick)
            if true_value != 0:
                true += 1  # Count the true pick
            if pred_value != 0:
                positive += 1  # Count the predicted pick

            # Calculate the residual difference (scaled by dt)
            if true_value != 0 and pred_value != 0:
                diff = dt * (pred_value - true_value)
                # Check if the residual is within the tolerance window
                if torch.abs(diff) <= tol:
                    true_positive += 1
                    residual.append(diff.item())  # Add residual as a plain number
                # Append to respective residual list (P or S)
                if phase_name == "P":
                    diff_p.append(diff.item())
                elif phase_name == "S":
                    diff_s.append(diff.item())

        # Compute metrics (precision, recall, F1 score)
        metrics[phase_name] = calc_metrics(true_positive, positive, true)

        if print_metrics:
            print(f"{phase_name}-phase:")
            print(f"True={true}, Positive={positive}, True Positive={true_positive}")
            print(f"Precision={metrics[phase_name][0]:.3f}, Recall={metrics[phase_name][1]:.3f}, F1={metrics[phase_name][2]:.3f}")
            if len(residual) > 0:
                print(f"Residual mean={torch.mean(torch.tensor(residual)):.4f}, std={torch.std(torch.tensor(residual)):.4f}")

    return metrics, residual, diff_p, diff_s



def plot_residual(diff_p, diff_s, diff_ps, tol=10.0, dt=1.0, plt_title=""):
    box = dict(boxstyle='round', facecolor='white', alpha=1)
    text_loc = [0.07, 0.95]
    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1)
    plt.hist(diff_p, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.ylabel("Number of picks")
    plt.xlabel("Residual (s)")
    plt.xticks(np.linspace(-tol, tol, 5), [f"{x:.2f}" for x in np.linspace(-tol/100, tol/100, 5)])
    plt.text(text_loc[0], text_loc[1], "(i)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("P-phase")
    plt.subplot(1,3,2)
    plt.hist(diff_s, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.xlabel("Residual (s)")
    plt.xticks(np.linspace(-tol, tol, 5), [f"{x:.2f}" for x in np.linspace(-tol/100, tol/100, 5)])
    plt.text(text_loc[0], text_loc[1], "(ii)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("S-phase")
    plt.subplot(1,3,3)
    plt.hist(diff_ps, range=(-tol, tol), bins=int(2*tol/dt)+1, facecolor='b', edgecolor='black', linewidth=1)
    plt.xlabel("Residual (s)")
    plt.xticks(np.linspace(-tol, tol, 5), [f"{x:.2f}" for x in np.linspace(-tol/100, tol/100, 5)])
    plt.text(text_loc[0], text_loc[1], "(iii)", horizontalalignment='left', verticalalignment='top',
            transform=plt.gca().transAxes, fontsize="small", fontweight="normal", bbox=box)
    plt.title("PS-phase")
    plt.tight_layout()
    if plt_title!="":
        plt.savefig(plt_title+".jpg", dpi=500, bbox_inches='tight', facecolor=None)
        # plt.savefig(plt_title+".pdf")
    plt.show()



################################## MODELS #####################################


class PickerDecoder(nn.Module):
    def __init__(self, num_channels=384, num_classes=3, upsample_scales=[8, 8, 5], dropout_rate=0.2):
        super(PickerDecoder, self).__init__()
        
        # Decoder: Upsampling blocks
        self.conv1=nn.ConvTranspose1d(num_channels, 256, kernel_size=3, stride=upsample_scales[0], padding=0,  output_padding=1)
        self.bn1=nn.BatchNorm1d(256)
        self.relu=nn.ReLU(inplace=True)
        self.dropout=nn.Dropout(p=dropout_rate)
            
        self.conv2=nn.ConvTranspose1d(256, 64, kernel_size=3, stride=upsample_scales[1], padding=1)
        self.bn2=nn.BatchNorm1d(64)

        self.conv3=nn.ConvTranspose1d(64, 16, kernel_size=7, stride=upsample_scales[2])
        self.bn3=nn.BatchNorm1d(16)
        self.adjust_length = nn.Conv1d(16, 16, kernel_size=13, padding=0)
                
        self.layer_out=nn.Conv1d(16, num_classes, kernel_size=8)  
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.dropout(self.relu(self.bn1(self.conv1(x))))
        x = self.dropout(self.relu(self.bn2(self.conv2(x))))
        x = self.dropout(self.relu(self.bn3(self.conv3(x))))
        x = self.layer_out(x)
       
        return  x 
    







class NewEncodecModel(nn.Module):
    def __init__(self, random_model=False, dropout_rate=0.2):
        super(NewEncodecModel, self).__init__()

        configuration = EncodecConfig()
        net_E =  EncodecModel(configuration)
        net_N =  EncodecModel(configuration)
        net_Z =  EncodecModel(configuration)

        if not random_model:
            net_E.load_state_dict(torch.load("../models/STEAD_ch0.pth", map_location='cuda:0'))
            net_N.load_state_dict(torch.load("../models/STEAD_ch1.pth", map_location='cuda:0'))
            net_Z.load_state_dict(torch.load("../models/STEAD_ch2.pth", map_location='cuda:0'))

        self.encoder_E = net_E.encoder
        self.encoder_N = net_N.encoder
        self.encoder_Z = net_Z.encoder
        
        # Identify layers for skip connections
        self.skip_layer_indices = [0, 3, 6, 9, 12, 15]  
        self.encoder_skip_outputs = {}  

        # Updated decoder with all skip connections
        self.decoder = nn.ModuleList([
            nn.Conv1d(128*6, 512, kernel_size=1, stride=1),
            nn.Dropout(p=dropout_rate),
            nn.LSTM(512, 512, num_layers=2),
            nn.Dropout(p=dropout_rate),
            nn.ELU(alpha=1.0),

            nn.ModuleList([
                nn.ELU(alpha=1.0),
                nn.Conv1d(3*512 + 512, 512, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate),  
                nn.ELU(alpha=1.0),
                nn.Conv1d(512, 512, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate)
            ]),

            nn.ConvTranspose1d(512, 256, kernel_size=12, stride=7), 
            nn.Dropout(p=dropout_rate),
            nn.Dropout(p=dropout_rate),
            nn.ModuleList([
                nn.ELU(alpha=1.0),
                nn.Conv1d(3*256 + 256, 256, kernel_size=1, stride=1),  
                nn.Dropout(p=dropout_rate),
                nn.ELU(alpha=1.0),
                nn.Conv1d(256, 256, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate)
            ]),

            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=5),  
            nn.Dropout(p=dropout_rate),
            nn.ModuleList([
                nn.ELU(alpha=1.0),
                nn.Conv1d(3*128 + 128, 128, kernel_size=1, stride=1),  
                nn.Dropout(p=dropout_rate),
                nn.ELU(alpha=1.0),
                nn.Conv1d(128, 128, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate)
            ]),

            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4), 
            nn.Dropout(p=dropout_rate),
            nn.ModuleList([
                nn.ELU(alpha=1.0),
                nn.Conv1d(3*64 + 64, 64, kernel_size=1, stride=1),  
                nn.Dropout(p=dropout_rate),
                nn.ELU(alpha=1.0),
                nn.Conv1d(64, 64, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate)
            ]),

            nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2), 
            nn.Dropout(p=dropout_rate), 
            nn.ModuleList([
                nn.ELU(alpha=1.0),
                nn.Conv1d(32, 32, kernel_size=1, stride=1), 
                nn.Dropout(p=dropout_rate),  
                nn.ELU(alpha=1.0),
                nn.Conv1d(32, 32, kernel_size=1, stride=1),
                nn.Dropout(p=dropout_rate)
            ]),

            nn.ConvTranspose1d(32, 3, kernel_size=1, stride=1),
            nn.Dropout(p=dropout_rate)
        ])
        self.softmax = nn.Softmax(dim=2) 
        self.decoder[2].flatten_parameters() 

    def _skip_connection_block(self, in_channels, out_channels, skip_connection):
        block = nn.ModuleList([
            nn.ELU(alpha=1.0),
            nn.Conv1d(in_channels + in_channels, out_channels, kernel_size=3, stride=1),  
            nn.ELU(alpha=1.0),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1),  
        ])
        block.is_skip_connection_block = True  
        return block

    def _combine_skip_connections(self):
        combined_skips = {}
        for skip_key in self.encoder_skip_outputs.keys():  
            channel_skips = [self.encoder_skip_outputs[skip_key] for _ in range(3)]  
            combined_skip = torch.cat(channel_skips, dim=1)  
            combined_skips[skip_key] = combined_skip
        combined_skips_reversed = {15-k: v for k, v in combined_skips.items()}
        return combined_skips_reversed
    
    def _encode_with_skip(self, x, encoder):
        output = x
        for i, layer in enumerate(encoder.layers):
            output = layer(output)
            if i in self.skip_layer_indices:
                self.encoder_skip_outputs[i] = output
            if isinstance(layer, nn.LSTM):  
                output = output[0]  
        return output
		

    def forward(self, x):
        x_channels = [x[:, i:i+1, :] for i in range(3)]  
        encoded_channels = []
        self.encoder_skip_outputs = {}  
        c=0
        for channel in x_channels:
            self.encoder_skip_outputs = {}  
            if c == 0:
                encoded_channel = self._encode_with_skip(channel, self.encoder_Z)
            elif c == 1:
                encoded_channel = self._encode_with_skip(channel, self.encoder_E)
            elif c == 2:
                encoded_channel = self._encode_with_skip(channel, self.encoder_N)

            encoded_channels.append(encoded_channel)
            c+=0

        combined_encoded = torch.cat(encoded_channels, dim=1)
        combined_skip_connections = self._combine_skip_connections()

        output = combined_encoded
        skip_idx = len(combined_skip_connections)-1
        
        for i, layer in enumerate(self.decoder):
            if i==0:
                skip_key = list(combined_skip_connections.keys())[skip_idx]
                skip_value = list(combined_skip_connections.values())[skip_idx] 
                output = torch.cat((output, skip_value), dim=1)  
                skip_idx-=1   

            if isinstance(layer, nn.ModuleList):  # Check if layer is a ModuleList (skip connection block)                
                output = layer[0](output)  # ELU
                skip_key = list(combined_skip_connections.keys())[skip_idx]
                skip_value = list(combined_skip_connections.values())[skip_idx] 
                if i==18:
                    output = layer[1](output)                     
                else:
                    output = layer[1](torch.cat((output, skip_value), dim=1))  # Conv1d with skip                        
                    
                output = layer[2](output)  
                output = layer[3](output)  
                skip_idx-=1
            elif isinstance(layer, nn.LSTM):  
                output = output.permute(2, 0, 1) 
                output, _ = layer(output) 
                output = output.permute(1, 2, 0)
                 
            else:            
                output = layer(output)
        return output
