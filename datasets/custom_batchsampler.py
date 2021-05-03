from torch.utils.data.sampler import Sampler

## one by one 
class CustomBatchSampler_Multi(Sampler):

    def __init__(self, sampler):
        for samp in sampler:
            if not isinstance(samp, Sampler):
                raise ValueError("sampler should be an instance of "
                                 "torch.utils.data.Sampler, but got sampler={}"
                                 .format(samp))
        self.samplers = sampler
        self.n_samples = [len(samp) for samp in self.samplers]
        self.sample_cnt = [0 for samp in self.samplers]
        self.iters = [iter(samp) for samp in self.samplers]
        
    def __iter__(self):       

        # for each iteration step
        for ii in range(len(self)):

            # if index is the even number
            if ii % 2 == 0:
                sampler_id = 0
            else:
                sampler_id = 1

            self.sample_cnt[sampler_id] += 1    # the nubmer of used sample. One sample per one iteration.
            if self.sample_cnt[sampler_id] > self.n_samples[sampler_id]: ## if exceeding the number of samples, reinitialize the sampler
                self.iters[sampler_id] = iter(self.samplers[sampler_id])
                self.sample_cnt[sampler_id] = 1

            batch = []

            ## starting index of the iterator
            if sampler_id is 0:
                prev_idx = 0
            else:
                prev_idx = self.n_samples[sampler_id-1]

            ## include a sample in the batch
            batch.append(next(self.iters[sampler_id]) + prev_idx)

            yield batch        

    def __len__(self):

        return len(self.samplers[0]) * 2