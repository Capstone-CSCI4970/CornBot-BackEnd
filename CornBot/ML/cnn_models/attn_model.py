import torch
import torch.nn as nn
from torchvision import transforms,models
import torch.nn.functional as F


class MyVgg(nn.Module):
  '''
  PreTrained VGG model(fixed Feature Extractor) with
  Attention Mechanism
  https://arxiv.org/pdf/1804.02391v2.pdf
  '''
  def __init__(self):
    super(MyVgg,self).__init__()
    vgg = models.vgg19_bn(pretrained=True)
    list_of_modules = list(next(vgg.children()))
    self.layer1 = nn.Sequential(*list_of_modules[:27])
    self.projector = nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False)
    self.attn1 =  LinearAttentionBlock(512)
    self.layer2 = nn.Sequential(*list_of_modules[27:40])
    self.attn2 = LinearAttentionBlock(512)
    self.layer3 = nn.Sequential(*list_of_modules[40:])
    self.attn3 = LinearAttentionBlock(512)
    #self.dense = nn.Conv2d(512,512, kernel_size=int(32/32), padding=0, bias=True)
    self.dense = nn.Conv2d(512,512, kernel_size=int(224/32), padding=0, bias=True)
    self.freeze()
    self.classify = nn.Linear(512*3,2,bias=True)
  
  def forward(self,x):
    l1 = self.layer1(x)
    l2 = self.layer2(l1)
    l3 = self.layer3(l2)
    g = self.dense(l3)
    c1, g1 = self.attn1(self.projector(l1), g)
    c2, g2 = self.attn2(l2, g)
    c3, g3 = self.attn3(l3, g)
    g = torch.cat((g1,g2,g3), dim=1)
    x = self.classify(g)
    return [x,c1,c2,c3]


  def freeze(self):
    for lay in [self.layer1,self.layer2,self.layer3]:      
      for p in lay.parameters():
        p.requires_grad = False
  

class LinearAttentionBlock(nn.Module):
    def __init__(self, in_features, normalize_attn=True):
        super(LinearAttentionBlock, self).__init__()
        self.normalize_attn = normalize_attn
        self.op = nn.Conv2d(in_channels=in_features, out_channels=1, kernel_size=1, padding=0, bias=False)
    def forward(self, l, g):
        N, C, W, H = l.size()
        c = self.op(l+g) # batch_sizex1xWxH
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        g = torch.mul(a.expand_as(l), l)
        if self.normalize_attn:
            g = g.view(N,C,-1).sum(dim=2) # batch_sizexC
        else:
            g = F.adaptive_avg_pool2d(g, (1,1)).view(N,C)
        return c.view(N,1,W,H), g

