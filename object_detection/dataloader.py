from PascalVOCDataset import PascalVOCDataset
import torch

""" Data Loaders
"""
def get_data_loader(config, transform_train=None, transform_test=None):
    # assert config.dataset == "cifar10" or config.dataset == "cifar100"
    
    if config.dataset == "pascal-voc":
        trainset = PascalVOCDataset(config.data_path,
                                    split='train',
                                    keep_difficult=config.keep_difficult)
        testset = PascalVOCDataset(config.data_path,
                                   split='test',
                                   keep_difficult=config.keep_difficult)

        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers,
            collate_fn=trainset.collate_fn, pin_memory=True
        )

        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config.test_batch, shuffle=False, num_workers=config.workers,
            collate_fn=testset.collate_fn, pin_memory=True
        )

    else:
        pass

    return train_loader, test_loader