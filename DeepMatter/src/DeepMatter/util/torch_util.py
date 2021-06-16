from torch.utils.data import Dataset, DataLoader

def get_n_params(model):
    '''

    Function that gets the number of parameters in a pytorch model

    Args:
        model (object): pytorch model

    Returns: number of parameters in model

    '''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


class Dataset_Generator(Dataset):
    """
    Function that generates data
    """

    def __init__(self,
                 model,
                 samples_per_epoch=100000,
                 size=1,
                 **kwargs):
        self.model = model
        self.samples_per_epoch = samples_per_epoch
        self.a = kwargs.get('a')
        self.b = kwargs.get('b')
        self.h = kwargs.get('h')
        self.k = kwargs.get('k')
        self.T = kwargs.get('T')
        self.sd = kwargs.get('sd')
        self.amp = kwargs.get('amp')
        self.fraction = kwargs.get('fraction')
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.size = size
        self.function = self.model(self.x, self.y
                                   sd=self.sd,
                                   fraction=self.fraction,
                                   amp=self.amp,
                                   a=self.a,
                                   b=self.b,
                                   h=self.h,
                                   k=self.k,
                                   T=self.T,
                                   size=self.size)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        input, params = self.function.sampler(device='cuda')

        return {'input': input, 'params': params}