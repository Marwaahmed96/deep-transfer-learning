import numpy as np
import h5py
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, data, options, patches, phase='train'):
        self.data= data
        self.phase = phase
        self.patches= patches
        self.options= options
        self.batch_size= options['batch_size']

    def __len__(self):
        return self.patches//self.batch_size if self.patches % self.batch_size == 0 else self.patches//self.batch_size+1

    def __getitem__(self, batch_index=1):

        x_data = []
        y_data = []
        # needed baches
        batch_size = self.batch_size
        #index =0
        for p in self.data:
            #index +=1
            #if index % 3 ==0:
            #    break
            patient_x_data = self.data[p]['X'][()]
            #patient_x_data = patient_x_data[:270, :]
            if self.phase != 'test':
                patient_y_data = self.data[p]['Y'][()]
                #patient_y_data = patient_y_data[:270]
            else:
                patient_y_data = None

            iter_start = 0
            while iter_start < patient_x_data.shape[0]:
                #print('patient:', p, iter_start)
                if iter_start + batch_size < patient_x_data.shape[0]:
                    iter_end = iter_start + batch_size
                    if len(x_data) > 0:
                        x_data = np.concatenate((x_data, patient_x_data[iter_start: iter_end]), axis=0)
                        if self.phase != 'test':
                            y_data = np.concatenate((y_data, patient_y_data[iter_start: iter_end]), axis=0)
                        iter_start = iter_end
                    else:
                        x_data = patient_x_data[iter_start: iter_end]
                        if self.phase != 'test':
                            y_data = patient_y_data[iter_start: iter_end]

                else:
                    x_data = patient_x_data[iter_start:]
                    if self.phase != 'test':
                        y_data = patient_y_data[iter_start:]

                x = x_data
                y = y_data

                if len(x_data) != self.batch_size:
                    #nedded batches
                    batch_size = self.batch_size - len(x_data)
                    iter_start += batch_size
                else:
                    batch_size = self.batch_size
                    iter_start += batch_size
                    x_data = []
                    y_data = []
                    #handle if last batch less than batch_size
                    x = np.stack(x).astype(np.float32)
                    if self.phase != 'test':
                        y = np.stack(y).astype(np.float32)
                        yield x, y
                    else:
                        yield x

        if len(x) < self.batch_size:
            x = np.concatenate((x, patient_x_data[:self.batch_size-len(x)]), axis=0)
            #print('last',x.shape)
            if self.phase != 'test':
                y = np.concatenate((y, patient_y_data[:self.batch_size-len(y)]), axis=0)
                yield x, y
            else:
                yield x
