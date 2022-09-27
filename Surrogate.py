import numpy as np
import torch
import matplotlib.pyplot as plt

compliance = np.loadtxt("data/compliance.txt", delimiter=', ')
compliance = torch.from_numpy(compliance).to(torch.float32)
indicator = np.loadtxt("data/indicator.txt", delimiter=', ')
indicator = torch.from_numpy(indicator).reshape((-1,1,50,50)).to(torch.float32)

mu = torch.mean(compliance)
std = torch.std(compliance)

indicator = indicator[compliance<1e3]
compliance = compliance[compliance<1e3]

#compliance = (compliance - mu)/std
# Rescaling
compliance = compliance / torch.max(compliance) * 10 # was macht dieses rescaling?
indicator = indicator*2 - 1 # rescale from 0 1 to -1 1

def initWeights(m):
    """Initialize weights of neural network with xavier initialization."""
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
#        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu')) # xavier somehow performs better than He
        m.bias.data.fill_(0.01)

class NN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        
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
        x = torch.nn.LeakyReLU()(self.linear2(x))

        return x
       
torch.manual_seed(2)
model = NN()
model.apply(initWeights)

lr = 1e-4
epochs = 10
clip = 2
batches = 10
batchsize = 1

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
cost_history = []
cost_batches = np.zeros((batches))


for epoch in range(epochs):
    
    for batch in range(batches):
        compliancePrediction = model(indicator[batch*batchsize:(batch+1)*batchsize])
        
        cost = torch.mean((compliancePrediction - compliance[batch*batchsize:(batch+1)*batchsize])**2)
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        cost_batches[batch] = cost.detach().numpy()
        
    cost_epoch = np.mean(cost_batches)
    cost_history.append(cost_epoch)
        
        
    
    
    if epoch % 10 == 0:
        # print("prediction")
        # print(compliancePrediction)
        # print("cost")
        # print(cost)
        # print("\n")
        print(f"Epoch: {epoch} \t Cost: {cost_epoch}")

print(model(indicator[:batches*batchsize]))



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
ax.plot(np.arange(epochs), cost_history)
plt.xlabel("Epochs")
plt.ylabel("Cost")












