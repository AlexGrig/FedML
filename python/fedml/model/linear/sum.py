import torch


class Sum(torch.nn.Module):
    def __init__(self):
        super(Sum, self).__init__()

    def forward(self, x):
        # try:
        outputs = torch.sum(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
