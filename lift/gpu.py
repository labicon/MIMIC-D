import torch
import torch.cuda

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


if __name__ == "__main__":
    main()