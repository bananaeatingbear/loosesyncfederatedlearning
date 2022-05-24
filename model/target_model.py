import torch 
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class TargetNet(nn.Module):

    def __init__(self,dataset,input_feature_number=0,output_feature_number=0):
        super(TargetNet, self).__init__()
        self.dataset = dataset
        self.input_feature_number = input_feature_number
        self.output_feature_number = output_feature_number
        
        if ('fashion_product' in self.dataset):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
            self.activation1 = nn.ReLU(inplace=True)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            self.activation2 = nn.ReLU(inplace=True)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            self.activation3 = nn.ReLU(inplace=True)
            self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            self.activation4 = nn.ReLU(inplace=True)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.activation5 = nn.ReLU(inplace=True)
            self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            if ('season' in self.dataset):
                self.linear1 = nn.Linear(256, 4)
            elif ('gender' in self.dataset):
                self.linear1 = nn.Linear(256, 2)

        if (self.dataset == 'retina'):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
            self.activation1 = nn.ReLU(inplace=True)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            self.activation2 = nn.ReLU(inplace=True)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            self.activation3 = nn.ReLU(inplace=True)
            self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            self.activation4 = nn.ReLU(inplace=True)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.activation5 = nn.ReLU(inplace=True)
            self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(1024, 4)

        if (self.dataset == 'gtsrb'):
	        self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
	        self.activation1 = nn.ReLU(inplace=True)
	        self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
	        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
	        self.activation2 = nn.ReLU(inplace=True)
	        self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
	        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
	        self.activation3 = nn.ReLU(inplace=True)
	        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
	        self.activation4 = nn.ReLU(inplace=True)
	        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
	        self.activation5 = nn.ReLU(inplace=True)
	        self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
	        self.flatten = nn.Flatten()
	        self.linear1 = nn.Linear(256, 43)
            
        if (self.dataset == 'intel'):
            self.conv1 = nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5)
            self.activation1 = nn.ReLU(inplace=True)
            self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            self.activation2 = nn.ReLU(inplace=True)
            self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            self.activation3 = nn.ReLU(inplace=True)
            self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            self.activation4 = nn.ReLU(inplace=True)
            self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            self.activation5 = nn.ReLU(inplace=True)
            self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            self.linear1 = nn.Linear(256, 6)

        if (self.dataset == 'cifar100'):
            self.model = models.densenet161()

        if (self.dataset == 'cifar10'):
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, padding=2),
                #nn.BatchNorm2d(32),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.layer2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=5, padding=2),
                #nn.BatchNorm2d(64),
                nn.Dropout2d(),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.fc1 = nn.Linear(8*8*64, 128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,10)

        if (self.dataset == 'mnist'):
            self.conv1 =  nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.drop1 = nn.Dropout2d()
            self.act1 = nn.ReLU()
            self.max1 =  nn.MaxPool2d(2)
            self.conv2 =  nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.drop2 = nn.Dropout2d()
                #nn.BatchNorm2d(32),
            self.act2 = nn.ReLU()
            self.max2 =  nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(7*7*64, 500)
            self.drop3 = nn.Dropout()
            self.fc2 = nn.Linear(500,10)

        if (self.dataset == 'sat6'):
            self.conv1 =  nn.Conv2d(4, 32, kernel_size=5, padding=2)
            self.drop1 = nn.Dropout2d()
            self.act1 = nn.ReLU()
            self.max1 =  nn.MaxPool2d(2)
            self.conv2 =  nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.drop2 = nn.Dropout2d()
            self.act2 = nn.ReLU()
            self.max2 =  nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(7*7*64, 500)
            self.drop3 = nn.Dropout()
            self.fc2 = nn.Linear(500,6)


        if (self.dataset == 'fashion_mnist'):
            self.conv1 =  nn.Conv2d(1, 32, kernel_size=5, padding=2)
            self.drop1 = nn.Dropout2d()
            self.act1 = nn.ReLU()
            self.max1 =  nn.MaxPool2d(2)
            self.conv2 =  nn.Conv2d(32, 64, kernel_size=5, padding=2)
            self.drop2 = nn.Dropout2d()
                #nn.BatchNorm2d(32),
            self.act2 = nn.ReLU()
            self.max2 =  nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(7*7*64, 500)
            self.drop3 = nn.Dropout()
            self.fc2 = nn.Linear(500,10)

            #self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=4, padding=5)
            #self.activation1 = nn.ReLU(inplace=True)
            #self.maxpooling1 = nn.MaxPool2d(kernel_size=2, stride=2)
            #self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
            #self.activation2 = nn.ReLU(inplace=True)
            #self.maxpooling2 = nn.MaxPool2d(kernel_size=2, stride=2)
            #self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
            #self.activation3 = nn.ReLU(inplace=True)
            #self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
            #self.activation4 = nn.ReLU(inplace=True)
            #self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
            #self.activation5 = nn.ReLU(inplace=True)
            #self.maxpooling5 = nn.MaxPool2d(kernel_size=2, stride=2)
            #self.flatten = nn.Flatten()
            #self.linear1 = nn.Linear(256, 10)

        if (self.dataset == 'adult'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,output_feature_number)
        
        if (self.dataset == 'texas'):
            #self.fc = nn.Linear(input_feature_number,2048)
            self.fc1 = nn.Linear(input_feature_number,1024)
            self.fc2 = nn.Linear(1024,512)
            self.fc3 = nn.Linear(512,256)
            self.fc4 = nn.Linear(256,100)

        if (self.dataset == 'purchase'):
            self.fc1 = nn.Linear(input_feature_number,1024)
            self.fc2 = nn.Linear(1024,512)
            self.fc3 = nn.Linear(512,256)
            self.fc4 = nn.Linear(256,100)

        if (self.dataset == 'titanic'):
            self.fc1 = nn.Linear(input_feature_number,128)
            self.dropout = nn.Dropout()
            self.fc2 = nn.Linear(128,output_feature_number)
        
    def forward(self, x):
        
        if (self.dataset == 'mnist'  or self.dataset == 'sat6' or self.dataset == 'fashion_mnist'): ## or self.dataset == 'fashion_mnist'
            out = self.conv1(x)
            out = self.drop1(out)
            out = self.act1(out)
            out = self.max1(out)

            #print (out.size())

            out = self.conv2(out)
            out = self.drop2(out)
            out = self.act2(out)
            out = self.max2(out)

            #print (out.size())
            
            out = self.flatten(out)
            
            #print (out.size())

            out = self.fc1(out)
            out = self.drop3(out)
            out = self.fc2(out)

        if (self.dataset == 'cifar10' or self.dataset == 'cifar100'):
            out = self.layer1(x)
            out = self.layer2(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            #out = F.softmax(out, dim=1)

        if (self.dataset == 'adult' or self.dataset == 'titanic'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = self.dropout(out)
            out = self.fc2(out)
            #out = F.log_softmax(out,dim=1)

        if (self.dataset == 'texas'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = F.tanh(out)
            out = self.fc2(out)
            out = F.tanh(out)
            out = self.fc3(out)
            out = F.tanh(out)
            out = self.fc4(out)
            out = F.tanh(out)

        if (self.dataset == 'purchase'):
            out = x.view(x.size(0),-1)
            out = self.fc1(out)
            out = F.tanh(out)
            out = self.fc2(out)
            out = F.tanh(out)
            out = self.fc3(out)
            out = F.tanh(out)
            out = self.fc4(out)
            out = F.tanh(out)

        if (self.dataset == 'retina'):
            x = self.conv1(x)
            x = self.activation1(x)
            x = self.maxpooling1(x)
            x = self.conv2(x)
            x = self.activation2(x)
            x = self.maxpooling2(x)
            x = self.conv3(x)
            x = self.activation3(x)
            x = self.conv4(x)
            x = self.activation4(x)
            x = self.conv5(x)
            x = self.activation5(x)
            x = self.maxpooling5(x)
            x = self.flatten(x)
            #print (x.size())
            out = self.linear1(x)

        if (self.dataset == 'gtsrb'):
	        x = self.conv1(x)
	        x = self.activation1(x)
	        x = self.maxpooling1(x)
	        x = self.conv2(x)
	        x = self.activation2(x)
	        x = self.maxpooling2(x)
	        x = self.conv3(x)
	        x = self.activation3(x)
	        x = self.conv4(x)
	        x = self.activation4(x)
	        x = self.conv5(x)
	        x = self.activation5(x)
	        x = self.maxpooling5(x)
	        x = self.flatten(x)
	        # print (x.size())
	        out = self.linear1(x)

        if ('fashion_product' in self.dataset):
            x = self.conv1(x)
            x = self.activation1(x)
            x = self.maxpooling1(x)
            x = self.conv2(x)
            x = self.activation2(x)
            x = self.maxpooling2(x)
            x = self.conv3(x)
            x = self.activation3(x)
            x = self.conv4(x)
            x = self.activation4(x)
            x = self.conv5(x)
            x = self.activation5(x)
            x = self.maxpooling5(x)
            x = self.flatten(x)
            #print (x.size())
            out = self.linear1(x)
            
        if (self.dataset == 'intel'): # or self.dataset =='fashion_mnist'
            x = self.conv1(x)
            x = self.activation1(x)
            x = self.maxpooling1(x)
            #print (x.size())
            x = self.conv2(x)
            x = self.activation2(x)
            x = self.maxpooling2(x)
            #print (x.size())
            x = self.conv3(x)
            x = self.activation3(x)
            #print (x.size())
            x = self.conv4(x)
            x = self.activation4(x)
            #print (x.size())
            x = self.conv5(x)
            x = self.activation5(x)
            #print (x.size())
            x = self.maxpooling5(x)
            x = self.flatten(x)
            #print (x.size())
            out = self.linear1(x)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class celeba_model(nn.Module):
    def __init__(self):
        super(celeba_model, self).__init__()
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 32, 3),  # 3, 256, 256
            nn.MaxPool2d(2),  # op: 16, 127, 127
            nn.ReLU(),  # op: 64, 127, 127
        )
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(32, 64, 3),  # 64, 127, 127
            nn.MaxPool2d(2),  # op: 128, 63, 63
            nn.ReLU()  # op: 128, 63, 63
        )
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3),  # 128, 63, 63
            nn.MaxPool2d(2),  # op: 256, 30, 30
            nn.ReLU()  # op: 256, 30, 30
        )
        self.Linear1 = nn.Linear(512, 100)
        self.Linear2 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.ConvLayer1(x)
        #print (x.size())
        x = self.ConvLayer2(x)
        #print (x.size())
        x = self.ConvLayer3(x)
        #print (x.size())
        #x = self.ConvLayer4(x)
        x = x.view(x.size(0), -1)
        #print (x.size())
        x = self.Linear1(x)
        x = self.Linear2(x)
        #x = self.Linear3(x)
        return x