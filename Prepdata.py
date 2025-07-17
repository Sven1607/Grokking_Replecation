import numpy as np
import torch
import random
import devices


class Prep:
    device = devices.device.choose_dev()
    SEED = 42
    
    def set_seed(seed_value=0):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # Prime number for modular addition
    P = 53

    # Create the dataset
    set_seed(SEED)
    data = []
    for i in range(P):
        for j in range(P):
            data.append([i,j,(i+j)%P])
    data = np.array(data)

    # Split into train and test
    TRAIN_FRACTION = 0.5
    np.random.shuffle(data)
    train_data = data[:int(len(data) * TRAIN_FRACTION)]
    test_data = data[int(len(data) * TRAIN_FRACTION):]

    # Convert to tensors and create dataloaders with batch size
    BATCH_SIZE = 64
    train_data = torch.tensor(train_data, dtype=torch.long, device=device)
    test_data = torch.tensor(test_data, dtype=torch.long, device=device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)