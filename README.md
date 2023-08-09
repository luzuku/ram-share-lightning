
## RAM-share reproduction in Lightning

Reproduction of https://github.com/ppwwyyxx/RAM-multiprocess-dataloader in Lightning.
Worker PIDs are shared through a file.
The script will log `total_pss (GB)` with the progress bar.

[This blog post](https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/)
explains everything.

### Dependencies
* Python >= 3.7
* Linux
* PyTorch >= 1.10
* PyTorch Lightning >= 2.0


### Modes
Please handle devices through `CUDA_VISIBLE_DEVICES`.
* Naive: `CUDA_VISIBLE_DEVICES=0,1 python main-lightning.py --mode naive --num_workers 4 --strategy ddp`
* Numpy: `CUDA_VISIBLE_DEVICES=0,1 python main-lightning.py --mode numpy --num_workers 4 --strategy ddp`
* Torch: `CUDA_VISIBLE_DEVICES=0,1 python main-lightning.py --mode torch --num_workers 4 --strategy ddp`
* Torch-Multi-GPU: `CUDA_VISIBLE_DEVICES=0,1 python main-lightning.py --mode ddp --num_workers 4 --strategy ddp`
