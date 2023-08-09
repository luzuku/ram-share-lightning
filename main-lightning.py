#!/usr/bin/env python
import os
import time
import torch
import plotext as plt
import lightning as L

from argparse import ArgumentParser

import serialize
import common
import comm

from boring_classes import BoringModel


class SharingModel(BoringModel):
    def __init__(self, mode="naive", num_workers=1, pin_memory=False):
        super().__init__()
        self.mode = mode.lower()
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.worker_pids = None  # Will read from file
        self.main_pids = None  # Will sync with all_gather
        self.pids = None
        self.worker_pids_file = "worker_pids.txt"
        
        assert self.mode in ("naive", "numpy", "torch", "ddp")

    def worker_init_fn(self, worker_id):
        pid = os.getpid()
        with open(self.worker_pids_file, 'a') as file:
            file.write(str(pid) + '\n')
        print(f"Setup Rank: {self.global_rank} Worker-ID: {worker_id:02}, PID: {pid}")

    def get_total_pss(self):
        if self.worker_pids is None:
            with open(self.worker_pids_file, 'r') as file:
                self.worker_pids = [int(line.strip()) for line in file]
            # lightning calls iter(dataloader) twice for some reason
            valid_pids = len(self.worker_pids) // 2
            main_pids = self.all_gather(torch.as_tensor(self.main_pid, device=self.device))
            self.pids = main_pids.int().tolist() + self.worker_pids[valid_pids:]
        pss = sum(common.get_mem_info(pid)["pss"] for pid in self.pids)
        return pss

    def step(self, batch):
        output = self(batch)
        pss = self.get_total_pss() / 1024 / 1024 / 1024  # GB
        self.log("total_pss", pss, prog_bar=True)
        return self.loss(output)
        
    def train_dataloader(self):
        

        if self.mode == "naive":
            lst = torch.rand(10000*64*2, 32).tolist()
        elif self.mode == "numpy":
            lst = torch.rand(10000*64*2, 32).tolist()
            lst = serialize.NumpySerializedList(lst)
        elif self.mode == "torch":
            lst = torch.rand(10000*64*2, 32).tolist()
            lst = serialize.TorchSerializedList(lst)
        elif self.mode == "ddp":
            comm.create_local_process_group(self.trainer.world_size)
            comm.synchronize()
            serialize.local_broadcast_process_authkey()
            lst = []
            if self.global_rank == 0:
                lst = torch.rand(10000*64*2, 32).tolist()
            lst = serialize.TorchShmSerializedList(lst)
            
        ds = common.DatasetFromList(lst)

        if os.path.exists(self.worker_pids_file):
            os.remove(self.worker_pids_file)
            
        self.worker_pids = None
        self.main_pid = os.getpid()
        print(f"Called train_dataloader on {self.global_rank} with pid {self.main_pid}")
        
        return torch.utils.data.DataLoader(
            ds,
            num_workers=self.num_workers,
            batch_size=64,
            pin_memory=self.pin_memory,
            worker_init_fn=self.worker_init_fn,
        )

if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--mode", type=str, default="naive")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=int, default=0)
    parser.add_argument("--strategy", type=str, default="ddp")

    args = parser.parse_args()
    
    model = SharingModel(
        mode=args.mode,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory != 0,
    )

    # You should handle devices through the environment variable
    # e.g. CUDA_VISIBLE_DEVICES=0,1
    
    trainer = L.Trainer(
        logger=False,
        strategy=args.strategy,
        max_epochs=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    trainer.fit(model=model)
    