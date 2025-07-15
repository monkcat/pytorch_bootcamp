
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#NUM_WORKERS = os.cpu_count()
NUM_WORKERS = 0

def create_dataloaders(train_dir: str, 
                      test_dir: str,
                      transform: transforms.Compose,
                      batch_size: int,
                      num_workers: int=NUM_WORKERS):
    """
    학습용 & 테스트용 데이터로더 만드는 함수

    학습용 데이터와 테스트용 데이터가 있는 디렉토리 경로를 받아서 파이토치 데이터셋과 데이터로더로 변환

    Args:
        train_dir : 훈련용 데이터 디렉토리 경로
        test_dir : 
        transform :
        batch_size :
        num_workers :
    """

    train_data = datasets.ImageFolder(train_dir, transform = transform)
    test_data = datasets.ImageFolder(test_dir, transform  = transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = num_workers,
        pin_memory = True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size = batch_size,
        shuffle = False,
        num_workers = num_workers,
        pin_memory = True
    )

    return train_dataloader, test_dataloader, class_names
