import torch


class LinearRegression_batch(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression_batch, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # try:
        outputs = self.linear(x)
        # except:
        #     print(x.size())
        #     import pdb
        #     pdb.set_trace()
        return outputs
