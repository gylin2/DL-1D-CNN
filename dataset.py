import os
import torch
from torch.utils.data import Dataset
torch.set_default_tensor_type(torch.FloatTensor)
import librosa

class VPCID(Dataset):
    def __init__(self, path_to_audio, part='train', case='case0', augment=None):
        self.path_to_audio = os.path.join(path_to_audio, 'flac_16k')
        self.path_to_protocol = os.path.join(path_to_audio, 'protocols_format', case)
        self.part = part
        self.augment = augment
        protocol = os.path.join(self.path_to_protocol, 'VPCID_2s.' + self.part + '.txt')
        self.label = {"voip": 1, "mobile": 0}
        self.devices = {"mobile": 0, "Wetalk": 1, "Start": 2, "WeiweiMultiparty": 3,
                        "Vhua": 4, "Skype": 5, "Uwewe": 6, "Ailiao": 7, "Alicall": 8}
        
        self.caller_locations = {"a": 0, "b": 1, "c": 2, "d": 3}
        self.callee_locations = {"e": 0, "f": 1}

        with open(protocol, 'r') as f:
            audio_info = [info.strip().split() for info in f.readlines()]
            self.all_info = audio_info
            print(len(self.all_info))

    def __len__(self):
        return len(self.all_info)

    def __getitem__(self, idx):
        filename, label, devices, speakers, caller_location, callee_location, format_type = self.all_info[idx]
        wav, sr = librosa.load(os.path.join(self.path_to_audio, filename + '.flac'), sr=16000)
        wav = torch.from_numpy(wav).unsqueeze(0)
        
        return wav.squeeze(), self.label[label]


if __name__ == "__main__":
    path_to_audio = ''