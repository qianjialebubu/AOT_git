from .dataset import InpaintingData

from torch.utils.data import DataLoader


def sample_data(loader): 
    while True:
        # print(loader)
        for batch in loader:
            yield batch


def create_loader(args):

    dataset = InpaintingData(args)
    # print(dataset)
    data_loader = DataLoader(
        dataset, batch_size=args.batch_size//args.world_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # print("create_loader succeeded!")
    return sample_data(data_loader)

