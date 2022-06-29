from builtins import super
from collections import OrderedDict

import torch.nn as nn


class ModelCNN(nn.Module):

    def __init__(self,labels,word_embedding_size,batch_size,weights_tensor):
        super(ModelCNN, self).__init__()
        self.word_embedding_size=word_embedding_size
        self.batch_size=batch_size

        self.embedding_layer=nn.Sequential(
            nn.Embedding.from_pretrained(embeddings=weights_tensor,freeze=True)
        )

        self.convnet=nn.Sequential(OrderedDict([
            ('c1',nn.Conv1d(self.word_embedding_size,32,5)),#128
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool1d(5)),
            ('c2', nn.Conv1d(32, 32, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool1d(5)),
            ('c3', nn.Conv1d(32, 64, 5)),
            ('relu3', nn.ReLU()),
            ('maxpool3', nn.MaxPool1d(35))
        ]))
        # self.dropout=nn.Dropout(p=0.2) # For Dropout regularization
        self.fc = nn.Sequential(OrderedDict([
            ('f4',nn.Linear(64,64)),
            ('relu4', nn.ReLU()),
            ('f5', nn.Linear(64, labels)),
            ('sig5', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, x):
        out=self.embedding_layer(x)
        out.transpose_(1,2)
        out=self.convnet(out)
        out=out.view(x.size(0),-1)
        self.fc.float() #Only use in CUDA
        out=self.fc(out)
        # out = self.dropout(self.fc(out))
        return out

