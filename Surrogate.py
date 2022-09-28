import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print("cuda available?: ", torch.cuda.is_available(), ", version: ", torch.version.cuda)



print(f" Device: {device}")
compliance = np.loadtxt("data/compliance.txt", delimiter=', ')
compliance = torch.from_numpy(compliance).to(torch.float32).to(device)
indicator = np.loadtxt("data/indicator.txt", delimiter=', ')
indicator = torch.from_numpy(indicator).reshape((-1,1,50,50)).to(torch.float32).to(device)

mu = torch.mean(compliance)
std = torch.std(compliance)

indicator = indicator[compliance<1e3]
compliance = compliance[compliance<1e3]

#compliance = (compliance - mu)/std
# Rescaling
comp_max = torch.max(compliance)
compliance = compliance / comp_max * 10 
indicator = indicator*2 - 1 # rescale from 0 1 to -1 1






def reverse_scaling(compliance, comp_max):
    return compliance/10*comp_max

def initWeights(m):
    """Initialize weights of neural network with xavier initialization."""
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
#        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu')) # xavier somehow performs better than He
        m.bias.data.fill_(0.01)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=1)
        
#        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
#        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#        
#        self.conv7 = torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
#        self.conv8 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#        
#        self.conv9 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
#        self.conv10 = torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        
        self.linear1 = torch.nn.Linear(2304, 512)
        self.linear2 = torch.nn.Linear(512, 1)
        
    def forward(self, x):
        x = torch.nn.LeakyReLU()(self.conv1(x))
        x = torch.nn.LeakyReLU()(self.conv2(x))
        x = self.pool(x)
    
        x = torch.nn.LeakyReLU()(self.conv3(x))
        x = torch.nn.LeakyReLU()(self.conv4(x))
        x = self.pool(x)
        
#        fig, ax = plt.subplots(2,5, figsize=(10,4))
#        for i in range(2):
#            for k in range(5):
#                ax[i, k].imshow(x[i,k].detach())
#                plt.show()
#        
#        x = torch.nn.ReLU()(self.conv5(x))
#        x = torch.nn.ReLU()(self.conv6(x))
#        x = self.pool(x)
#
#        x = torch.nn.ReLU()(self.conv7(x))
#        x = torch.nn.ReLU()(self.conv8(x))
#        x = self.pool(x)
#        
#        x = torch.nn.ReLU()(self.conv9(x))
#        x = torch.nn.ReLU()(self.conv10(x))
#        x = self.pool(x)

        x = x.reshape((-1, 2304))
        x = torch.nn.LeakyReLU()(self.linear1(x))
        x = torch.nn.ReLU()(self.linear2(x))

        return x
       
torch.manual_seed(2)
model = NN()
model.apply(initWeights)
model.to(device)

lr = 1e-5
epochs = 200
clip = 2
#batches = 10
batchsize = 1
index_split = 150

np.random.seed(5)
shuffleIndex = np.linspace(0,len(compliance)-1,len(compliance), dtype=int)
np.random.shuffle(shuffleIndex)
compliance = compliance[shuffleIndex]
indicator = indicator[shuffleIndex]



train_dataloader=DataLoader(TensorDataset(indicator[:index_split], compliance[:index_split]), batch_size=batchsize, shuffle=True)
test_dataloader=DataLoader(TensorDataset(indicator[index_split:], compliance[index_split:]), batch_size=batchsize, shuffle=True)
#train_dataloader = DataLoader(TensorDataset(indicator, compliance), batch_size=batchsize, shuffle=False)
#test_dataloader = DataLoader((indicator, compliance), batch_size=batchsize, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
cost_history = []
cost_test_history = []
cost_batches = np.zeros((index_split))
cost_test_batches = np.zeros((compliance.size()[0]-index_split))



for epoch in range(epochs):
    i=0
    for indicator_i, compliance_i in train_dataloader:
        # device siwthing skipped as its not working on my laptop
        compliancePrediction = model(indicator_i)
        cost = torch.mean((compliancePrediction - compliance_i)**2)
        
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_batches[i] = cost.cpu().detach().numpy()
        i+=1
        
    i = 0    
    for indicator_i, compliance_i in test_dataloader:
        compliancePrediction = model(indicator_i)
        cost_test = torch.mean((compliancePrediction - compliance_i)**2)
        
        
        optimizer.zero_grad()
        cost_test.backward()
        optimizer.step()
        cost_test_batches[i] = cost.cpu().detach().numpy()
        i+=1
        
    
    # for batch in range(batches):
    #     compliancePrediction = model(indicator[batch*batchsize:(batch+1)*batchsize])
        
    #     cost = torch.mean((compliancePrediction - compliance[batch*batchsize:(batch+1)*batchsize])**2)
        
    #     optimizer.zero_grad()
    #     cost.backward()
    #     optimizer.step()
    #     cost_batches[batch] = cost.cpu().detach().numpy()
    cost_test_epoch = np.mean(cost_test_batches)    
    cost_epoch = np.mean(cost_batches)
    cost_history.append(cost_epoch)
    cost_test_history.append(cost_test_epoch)

        
        
    
    
    if epoch % 10 == 0:
        # print("prediction")
        # print(compliancePrediction)
        # print("cost")
        # print(cost)
        # print("\n")
        print(f"Epoch: {epoch} \t Cost: {cost_epoch} \t test_Cost: {cost_test_epoch}")

print("Example predictions")
print(reverse_scaling(model(indicator[:batches*batchsize]), comp_max))
print("Ground truths")
print(reverse_scaling(compliance[:batches*batchsize], comp_max))





#print("transformed prediction")
#compliancePrediction_ = compliancePrediction*std + mu
#print(compliancePrediction_)








# import matplotlib.pyplot as plt
# #
# for i in range(2):
#     fig, ax = plt.subplots()
#     ax.imshow(indicator[i,0])
#     plt.show()


fig, ax = plt.subplots()
ax.plot(np.arange(epochs), cost_history, label="Training loss")
ax.plot(np.arange(epochs), cost_test_history, label="Test loss")

plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.grid(True)
plt.label












