import chainer
import numpy as np


def my_convertor(data, device=None):
    # data is tuple of numpy array
    #print len(data), type(data)
    #print len(data[0]), type(data[0])
    return tuple([chainer.cuda.to_gpu(d.astype('float32'), device) for d in data])

def batch_extend(tuple1, tuple2):
    L = len(tuple1)
    t1_list = []
    for i in range(L):
        t1_list.append( np.concatenate((tuple1[i], tuple2[i]), axis=0) )
    return tuple(t1_list)
        

class M_SerialIterator(chainer.iterators.SerialIterator):
    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.dataset[i:i_end]
        else:
            batch = self.dataset[self._order[i:i_end]]

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    np.random.shuffle(self._order)
                if rest > 0:
                    if self._order is None:
                        batch = batch_extend(batch, self.dataset[:rest])
                    else:
                        batch = batch_extend(batch, self.dataset[self._order[:rest]])
                self.current_position = rest
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch
    
    next = __next__

class MyIterator(chainer.dataset.Iterator):
    # iterator for training

    def __init__(self, dataset, batch_size, repeat=True, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self.epoch = 0
        self.epoch_detail = 0

    def __next__(self):
        return self.dataset.get_random_batch(self.batch_size)

    next = __next__


def validation(trainer=None, model=None, dataset=None):
    # evaluate voxel IoUs on all classes
    with chainer.configuration.using_config('train', False):
        ious = {}
        for class_id in dataset.class_ids:
            iou = 0
            for batch in dataset.get_all_batches_for_evaluation(100, class_id):
                batch = my_convertor(batch)
                iou += model.evaluate_iou(*batch).sum()
            iou /= dataset.num_data[class_id] * dataset.total_views
            ious['%s/iou_%s' % (dataset.set_name, class_id)] = iou
        ious['%s/iou' % dataset.set_name] = np.mean([float(v) for v in ious.values()])
        chainer.report(ious)


def lr_shift(trainer=None, optimizer=None, iterations=None, factor=0.1):
    if trainer.updater.iteration in iterations:
        optimizer.alpha *= factor
