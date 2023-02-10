import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution

#Feature Teacher
class Teacher_F(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size, num_layers, dropout):
        super(Teacher_F, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_size)))
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)

        self.fm1 = nn.Linear(in_size, hidden_size, bias=True)
        self.fm2 = nn.Linear(hidden_size, out_size, bias=True)
        self.dropout = dropout
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        feature = torch.where(torch.isnan(feature), self.imp_feat, feature)
        middle_representations = []

        h = self.fm1(feature)
        middle_representations.append(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = F.relu(h)
        h = self.fm2(h)
        middle_representations.append(h)

        return h, middle_representations

#Structure Teacher
class Teacher_S(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size, dropout, device):
        super(Teacher_S, self).__init__()

        self.tgc1 = GraphConvolution(in_size, hidden_size)
        self.tgc2 = GraphConvolution(hidden_size, out_size)
        self.dropout = dropout
        self.linear = nn.Linear(num_nodes, in_size, bias=True)
        self.pe_feat = torch.FloatTensor(torch.eye(num_nodes)).to(device)

    def forward(self, adj):

        middle_representations = []

        pe = self.linear(self.pe_feat)
        # pe = F.dropout(pe, self.dropout, training=self.training)

        h = self.tgc1(pe, adj)
        middle_representations.append(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.tgc2(h, adj)
        middle_representations.append(h)

        return h, middle_representations

#Student
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, nhid_feat, nhid_stru, tau=0.5):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.tau = tau

        self.feat2stu = torch.nn.Linear(nhid_feat, nhid)
        self.stru2stu = torch.nn.Linear(nhid_stru, nhid)

    def forward(self, adj, x):
        #imp[0]
        imp = torch.zeros([x.shape[0], x.shape[1]])
        x = torch.where(torch.isnan(x), imp, x)

        middle_representations = []
        h = self.gc1(x, adj)
        middle_representations.append(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.gc2(h, adj)
        middle_representations.append(h)

        return h, middle_representations

    #contrast loss
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, z3: torch.Tensor,
             mean: bool = True):

        R_stu_1 = z1[0]
        R_fea_1 = self.feat2stu(z2[0])
        R_str_1 = self.stru2stu(z3[0])
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        str_stu_1 = self.semi_loss(R_stu_1, R_str_1)
        fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
        str_stu_1 = str_stu_1.mean() if mean else str_stu_1.sum()

        R_stu_2 = z1[1]
        R_fea_2 = z2[1]
        R_str_2 = z3[1]
        fea_stu_2 = self.semi_loss(R_stu_2, R_fea_2)
        str_stu_2 = self.semi_loss(R_stu_2, R_str_2)
        fea_stu_2 = fea_stu_2.mean() if mean else fea_stu_2.sum()
        str_stu_2 = str_stu_2.mean() if mean else str_stu_2.sum()

        loss_mid_fea = fea_stu_1 + fea_stu_2
        loss_mid_str = str_stu_1 + str_stu_2

        return loss_mid_fea, loss_mid_str

