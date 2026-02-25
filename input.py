import torch
from torch.utils.data import Dataset, DataLoader

class QuantumStateDataset(Dataset):
    def __init__(self, inputs_data, targets_data):
        """
        Qui passi i tuoi dati reali.
        inputs_data: Tensore di shape (N_samples, seq_len, d) - Reale
        targets_data: Tensore di shape (N_samples, seq_len, d) - Complesso
        """
        self.inputs = inputs_data
        self.targets = targets_data
        
        # Un check di sicurezza (best practice)
        assert len(self.inputs) == len(self.targets), "Mismatch tra input e target!"

    def __len__(self):
        # Ritorna il numero totale di campioni nel dataset
        return len(self.inputs)

    def __getitem__(self, idx):
        # Estrae un singolo campione. Il DataLoader chiamerà questo metodo
        # per assemblare i batch in automatico.
        x = self.inputs[idx]
        y_target = self.targets[idx]
        
        return x, y_target