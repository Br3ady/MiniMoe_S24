import torch 
import config, Model

config = config.Config()
attn = Model.Attn(config)

x = torch.rand(5,8,16,1000)
out_shape = (x.size()[0],) + (x.size()[3],) + (x.size()[2]*x.size()[1],)
x1 = x.view(out_shape)
x2 = x.view(out_shape)

breakpoint()