import torch

class device:
    def choose_dev():
        dev = "cpu"
        if torch.cuda.is_available():
            dev = "cuda"
            print("Using GPU", torch.cuda.get_device_name(0))
        else:
            print("Using CPU")
            
            return dev

    device = choose_dev()
