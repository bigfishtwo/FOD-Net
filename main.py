import os
import numpy as np
from models.fodnet_model import fodnetModel
import argparse
from dipy.io.image import load_nifti
from dipy.segment.mask import bounding_box, crop
import torch
from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

class FODDataset(Dataset):
    def __init__(self, root_dir, path, phase):
        self.root_dir = root_dir
        self.folder_names = np.loadtxt(os.path.join(path, phase+'.txt'), dtype=str)
        stats = np.load(os.path.join(self.root_dir, 'stats.npz'))
        self.fodlr_mean = stats['means'].astype(np.float32)
        self.fodlr_std = stats['stds'].astype(np.float32)

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, index):
        file_name = os.path.join(self.root_dir, self.folder_names[index])
        # TODO: path
        fod_path = os.path.join(file_name, 'subsampled_WM_FODs.nii.gz')
        fodmt_path = os.path.join(file_name, 'WM_FODs.nii.gz')
        brain_mask_path = os.path.join(file_name, 'nodif_brain_mask.nii.gz')
        # load fod and brain_mask
        fodlr, fod_affine = load_nifti(fod_path)
        brain_mask, brain_mask_affine = load_nifti(brain_mask_path)
        fodmt, affine = load_nifti(fodmt_path)

        assert fodlr.shape[:3] == brain_mask.shape, 'Input fod and mask should have the same shape'

        # fodlr, fodmt = self.set_input(fodlr, fodmt, brain_mask, act_mask=None)

        return (fodlr, fodmt), (brain_mask, self.folder_names[index])


    def set_input(self, fodlr, fodmt, brain_mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        """
        brain_mask = np.asarray(brain_mask, dtype=np.float32)
        # Compute the bounding box of nonzero intensity voxels in the volume.
        mins, maxs = bounding_box(brain_mask)
        mins.append(None)
        maxs.append(None)

        # Crops the input volume
        fodlr = crop(fodlr, mins, maxs)
        fodmt = crop(fodmt, mins, maxs)

        # z-score
        fodlr = (fodlr - self.fodlr_mean)/self.fodlr_std

        return fodlr, fodmt


class DataGenerator:
    def __init__(self, batch_size, root, path, phase, shuffle=False, gpu_id=0):
        """
        generate batch-wise patch data
        :param batch_size: int, number of images in a batch
        :param root: str, root path of fod data folders
        :param path: str, path of split file and statisic file
        :param phase: str, 'train', 'val', 'test'
        :param shuffle: bool, if True, shuffle the sequence of data
        """
        self.batch_size = batch_size
        self.root = root
        self.phase = phase
        self.shuffle = shuffle
        self.path = path
        self.files = np.loadtxt(os.path.join(path, phase+'.txt'), dtype=str)

        if gpu_id != -1:
            self.device = torch.device('cuda:{}'.format(gpu_id))
        else:
            self.device = torch.device('cpu')

        self.fodlr, self.fodmt = None, None
        self.fod_shape = None
        self.visited = None

        stats = np.load(os.path.join(path, 'stats.npz'))
        self.fodlr_mean = stats['means'].astype(np.float32)
        self.fodlr_std = stats['stds'].astype(np.float32)
        self.fodlr_mean = torch.from_numpy(self.fodlr_mean).to(self.device)
        self.fodlr_std = torch.from_numpy(self.fodlr_std).to(self.device)

        self.patch_size = 9
        self.sequence = np.arange(0, len(self.files))
        self.idx = 0


    def batch_generator(self):
        if self.shuffle:
            np.random.shuffle(self.sequence)
            
        self.idx = 0
        self.fodlr, self.fodmt = self.preprocess(self.sequence[self.idx])
        self.fod_shape = self.fodlr.shape[:3]
        self.visited = np.zeros(self.fod_shape)

        while self.idx < len(self.files):
            patches = torch.tensor(np.zeros((self.batch_size, self.patch_size, self.patch_size, self.patch_size,45)),
                                   dtype=torch.float32).to(self.device)
            ground_truths = torch.tensor(np.zeros((self.batch_size, self.patch_size, self.patch_size, self.patch_size,45)),
                                         dtype=torch.float32).to(self.device)

            count = 0
            count, patches, ground_truths = self.get_batch(count, patches, ground_truths)

            if count < self.batch_size:
                self.idx += 1
                try:
                    self.fodlr, self.fodmt = self.preprocess(self.sequence[self.idx])
                except IndexError:
                    break
                self.fod_shape = self.fodlr.shape[:3]
                self.visited = np.zeros(self.fod_shape)
                count, patches, ground_truths = self.get_batch(count, patches, ground_truths)

            # patches = np.array(patches)
            # ground_truths = np.array(ground_truths)
            patches = patches.permute(0,4,1,2,3)
            ground_truths = ground_truths.permute(0,4,1,2,3)
            yield patches, ground_truths

    def preprocess(self, index):
        folder = os.path.join(self.root, self.files[index])
        # TODO: path
        fodlr, affine = load_nifti(os.path.join(folder,'subsampled_WM_FODs.nii.gz'))
        fodmt, affine = load_nifti(os.path.join(folder, 'WM_FODs.nii.gz'))
        brain_mask, affine = load_nifti(os.path.join(folder, 'nodif_brain_mask.nii.gz'))

        brain_mask = np.asarray(brain_mask, dtype=np.float32)
        # Compute the bounding box of nonzero intensity voxels in the volume.
        mins, maxs = bounding_box(brain_mask)
        mins.append(None)
        maxs.append(None)

        # Crops the input volume
        fodlr = crop(fodlr, mins, maxs)
        fodmt = crop(fodmt, mins, maxs)

        fodlr = torch.from_numpy(fodlr).to(self.device)
        fodmt = torch.from_numpy(fodmt).to(self.device)

        # z-score
        fodlr = (fodlr - self.fodlr_mean) / self.fodlr_std
        
        return fodlr, fodmt

    def get_batch(self, count, patches, ground_truths):
        for i in range(0, self.fod_shape[0] - self.patch_size + 1, self.patch_size//2):
            for j in range(0, self.fod_shape[1] - self.patch_size + 1, self.patch_size//2):
                for k in range(0, self.fod_shape[2] - self.patch_size + 1, self.patch_size//2):
                    if self.visited[i][j][k]:
                        continue
                    patch = self.fodlr[i:i + self.patch_size, j:j + self.patch_size,
                            k:k + self.patch_size,:]
                    gt = self.fodmt[i:i + self.patch_size, j:j + self.patch_size,
                         k:k + self.patch_size,:]
                    patches[count,:,:,:,:] = patch
                    ground_truths[count,:,:,:,:] = gt
                    count += 1
                    self.visited[i:i + self.patch_size//2, j:j + self.patch_size//2,
                            k:k + self.patch_size//2] = np.ones((self.patch_size//2, self.patch_size//2, self.patch_size//2))

                    if count == self.batch_size:
                        return count, patches, ground_truths

        return count, patches, ground_truths

    @property
    def file_name(self):
        return self.files[self.sequence[self.idx]]
    @property
    def file_path(self):
        return os.path.join(self.root, self.files[self.sequence[self.idx]])
    @property
    def num_iterd(self):
        return sum(sum(sum(self.visited)))

def nullable_string(val):
    if not val:
        return None
    return val


def build_argparser():
    DESCRIPTION = "Preprocessing fod-net."
    p = argparse.ArgumentParser(description=DESCRIPTION)
    p.add_argument('--root', type=str, default='/work/data/')
    p.add_argument('--stats_path', type=str, default='/home/code', help="The path to the other files")
    p.add_argument('--output_path', type=str, default='/home/code/output', help="The output path")
    p.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--num_epochs', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--resume', type=bool, default=False)
    p.add_argument('--test', type=bool, default=False)
    p.add_argument('--weights_path', type=str, default='/home/code/output/best_model.pth.tar')
    p.add_argument('--lr_decay', type=bool, default=False)

    return p

def setup():
    parser = build_argparser()
    args = parser.parse_args()
    gpu_ids = args.gpu_ids
    root = args.root
    path = args.stats_path
    weights_path = '/home/code/output/'+args.weights_path
    output_path = args.output_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.lr
    resume = args.resume
    on_test = args.test
    r_decay = args.lr_decay

    curr = time.strftime("%m%d_%H%M", time.gmtime(time.time()))
    try:
        os.makedirs(os.path.join(output_path, curr), exist_ok=True)
        output_path = os.path.join(output_path, curr)
        print('creat dir {}'.format(output_path))
    except:
        sys.exit('cannot create the dir')


    model = fodnetModel(gpu_ids, learning_rate)  # create a model given opt.model and other options

    if resume or on_test:
        checkpoint = torch.load(weights_path, map_location='cuda:'+gpu_ids)
        best_loss = checkpoint['best_loss']
        epoch = checkpoint['epoch']
        # learning_rate = checkpoint['lr']
        model.setup(weights_path)

        best_model = {'best_loss': best_loss,
                      'epoch': epoch,
                      'model': model.net.state_dict(),
                      'optimizer': model.optimizer.state_dict()
                      }
    else:
        best_loss = float('inf')
        best_model = {'best_loss': best_loss,
                      'epoch': -1,
                      'model': model.net.state_dict(),
                      'optimizer': model.optimizer.state_dict()
                      }
    torch.save(best_model, os.path.join(output_path, 'best_model.pth.tar'))


    train_loader = DataGenerator(batch_size, root, path, 'train', shuffle=True, gpu_id=gpu_ids) 
    val_loader = DataGenerator(batch_size, root, path, 'val', shuffle=True, gpu_id=gpu_ids)

    test_set = FODDataset(root, path, 'test')
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    loaders = {'train': train_loader,
               'val': val_loader,
               'test': test_loader}

    logs = {'train': [],
            'val': []}

    if on_test:
        phases = ['test']
        num_epochs = 1
    else:
        phases = ['train', 'val']

    beginning = time.time()


    for epoch in tqdm(range(num_epochs)):
        print('Epoch:{}'.format(epoch))
        print('-'*10)
        for phase in phases:
            running_loss = 0.0
            running_acc = 0.0
            count = 0
   
            if phase == "train":
                model.net.train()

            else:
                model.net.eval()
                if phase == "test":
                    model.setup(weights_path=os.path.join(output_path, 'best_model.pth.tar'))

            if phase =='test':
                dataloader = loaders[phase]
            else:
                dataloader = loaders[phase].batch_generator()

            for fodlr, fodmt in dataloader:
                if phase!='test':
                    loss,acc = model.suffering(phase, fodlr, fodmt)
       
                else:
                    flr, fmt = fodlr
                    brain_mask, file_name = fodmt
                    model.set_input(flr[0,:,:,:,:], fmt[0,:,:,:,:], brain_mask[0,:,:,:])
                    loss,acc = model.test(file_name[0], output_path)

                # print('count:', count,loss,acc)
                running_loss += loss
                running_acc += acc
                count += 1
   
            epoch_loss = running_loss / count
 
            print('Phase:{}, Loss:{:.3e}'.format(phase, epoch_loss))

            logs[phase].append(epoch_loss)

            # save best model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = {'best_loss': best_loss,
                              'epoch': epoch,
                              'model': model.net.state_dict(),
                              'optimizer': model.optimizer.state_dict(),
                              # 'lr': learning_rate
                              }
                torch.save(best_model, os.path.join(output_path, 'best_model.pth.tar'))
            if lr_decay and phase == 'train':
                model.scheduler.step(epoch_loss)

        checkpoint = {
            'best_loss': best_loss,
            'epoch': epoch,
            'model': model.net.state_dict(),
            'optimizer': model.optimizer.state_dict()
           
        torch.save(checkpoint, os.path.join(output_path, 'last_model.pth.tar'))
        if (epoch+1) % 10 == 0:
            torch.save(checkpoint, os.path.join(output_path, '{}th_model.pth.tar'.format(str(epoch+1))))


        np.savez(os.path.join(output_path, 'logs.npz'), train=logs['train'], val=logs['val'])

        plt.figure('train loss')
        plt.plot(logs['train'], label='train loss')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'train loss.png'))

        plt.figure('val loss')
        plt.plot(logs['val'], label='val loss')
        plt.legend()
        plt.savefig(os.path.join(output_path, 'val loss.png'))

    duration = time.time()-beginning
    print("Running time: {}".format(time.strftime("%H:%M:%S", time.gmtime(duration))))
    plt.show()


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

if __name__ == '__main__':
    setup()

