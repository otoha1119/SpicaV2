import torch.utils.data
from data.base_dataset import BaseDataset
from data import dicom_ctpcct_2x_dataset, dicom_ctpcct_2x_test_dataset

def find_dataset_using_name(dataset_name):
    if dataset_name == "dicom_ctpcct_2x":
        return dicom_ctpcct_2x_dataset.DicomCtpcct2xDataset
    elif dataset_name == "dicom_ctpcct_2x_test":
        return dicom_ctpcct_2x_test_dataset.DicomCtpcct2xTestDataset
    else:
        raise NotImplementedError(f"Unknown dataset_mode: {dataset_name}")

def get_option_setter(dataset_name):
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options

def create_dataset(opt):
    return CustomDatasetDataLoader(opt)

class CustomDatasetDataLoader():
    def __init__(self, opt):
        self.opt = opt
        dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = dataset_class(opt)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        return int(min(len(self.dataset), self.opt.max_dataset_size))

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data
