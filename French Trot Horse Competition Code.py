import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

drop_cols = ["RaceStartTime", "FoalingDate", "ClassRestriction"]

Standard = ["AgeRestriction", "Barrier", "ClassRestriction", "CourseIndicator", "DamID", "Distance", "FoalingCountry",
            "FoalingDate", "FrontShoes", "Gender", "GoingAbbrev", "GoingID", "HandicapDistance", "HandicapType",
            "HindShoes", "HorseAge", "HorseID", "JockeyID", "RaceGroup", "RaceID", "RacePrizemoney", "RaceStartTime",
            "RacingSubType", "Saddlecloth", "SexRestriction", "SireID", "StartType", "StartingLine", "Surface",
            "TrackID", "TrainerID", "WeightCarried", "WetnessScale"]

Performance = ["BeatenMargin", "Disqualified", "FinishPosition", "PIRPosition", "Prizemoney", "RaceOverallTime",
               "PriceSP", "NoFrontCover", "PositionInRunning", "WideOffRail"]

df = pd.read_csv('data.csv', index_col=0)
object_columns = df.select_dtypes(include=['object']).columns
df['RaceStartTime'] = pd.to_datetime(df['RaceStartTime'])
df_before_2021 = df[df['RaceStartTime'] < '2021-11-01']
df_after_2021 = df[df['RaceStartTime'] >= '2021-11-01']
df_before_2021_index = df_before_2021.index
df_after_2021_index = df_after_2021.index

# 1. convert all object data to strings
df = df.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
# 2. Analyze the values of each column of data
# for column in object_columns:
#     print(f"{column}: {df[column].nunique()}")
# 3. drop useless data columns
df = df.drop(drop_cols, axis=1)
object_columns = [col for col in object_columns if col not in drop_cols]
print(object_columns)
# Exclude columns from labels
object_columns = [col for col in object_columns if col not in Performance]
# 4. Normalization of numerical data
for column in Standard:
    if column not in drop_cols and column not in object_columns:
        df[column] = (df[column] - df[column].mean()) / df[column].std()
# 5. Unique heat coding for non-numeric columns
df = pd.get_dummies(df, columns=object_columns)
# 6. Delineation of training and test sets
df_train = df.loc[df_before_2021_index]
df_test = df.loc[df_after_2021_index]

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5)#, stop_words='english')
train = tfidf.fit_transform(df_train).toarray()
test =  tfidf.transform(df_test).toarray()

test_x = torch.tensor(test_x,dtype=torch.float32)
test_x1 = torch.tensor(test_x1,dtype=torch.float32)
####Transform the numpy to tensor####
print("Shape of train_x:",train_x.shape)
print("Shape of val_x:",val_x.shape)
print("Shape of test_x:",test_x1.shape)

class MLPNet(nn.Module):

    def __init__(self):
        super(MLPNet ,self).__init__()
        self.FL1 = nn.Linear(540,128)
        self.FL2 = nn.Linear(540,128)
        self.FL = nn.Sequential(
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,8),
            nn.ReLU(),
            nn.Linear(8,2)
        )

    def forward(self ,x ,x1):

        x = F.relu(self.FL1(x))
        x1 = F.relu(self.FL2(x1))
        h = torch.concat((x,x1),0)
        h = self.FL(h)
        h = torch.softmax(h,dim=0)
        return h

def evaluator(y_test, y_pred):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 0:
            TP +=1
        if y_test[i] == 0 and y_pred[i] == 1:
            FN +=1
        if y_test[i] == 1 and y_pred[i] == 0:
            FP +=1
        if y_test[i] == 1 and y_pred[i] == 1:
            TN +=1
    cm = np.array([[TP,FN],[FP,TN]])
    accu = (TP+TN)/(TP+TN+FP+FN)
    prec = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1 = 2 *((prec*recall)/(prec + recall))
    print(cm)
    print()
    print('Accuracy = {}\n'. format(accu))
    print('Precision = {}\n'. format(prec))
    print('Recall = {}\n'. format(recall))
    print('F1 score = {}\n'. format(f1))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 0.001
weight_decay = 5e-5
model = MLPNet()
model = model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_calculation = nn.CrossEntropyLoss().to(device)
epoch = 500
############Training MLP#######
for epoch in range(epoch):
    sum = 0
    Train_y = []
    Train_pred = []
    for x,x1,label in zip(train_x,train_x1,train_label):
        x = x.to(device)
        x1 = x1.to(device)
        label = label.to(device)
        pred = model(x,x1)
        pred = torch.reshape(pred,[1,-1])
        label = torch.reshape(label,[1,-1])
        loss = loss_calculation(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        pred = torch.argmax(pred, dim=1, keepdim=False)
        label = torch.argmax(label, dim=1, keepdim=False)
        Train_y.append(label)
        Train_pred.append(pred)
        for i, j in zip(pred, label):
            if (i == j):
                sum = sum + 1
    print("epoch:{},Train_acc:{}".format(epoch+1, sum/len(train_x)))
evaluator(Train_y,Train_pred)

print(device)

#####Validation#####
sum = 0
Val_y = []
Val_pred = []
with torch.no_grad():
    for x,x1,label in zip(val_x,val_x1,val_label):
        x = x.to(device)
        x1 = x1.to(device)
        label = label.to(device)
        pred = model(x, x1)
        pred = torch.reshape(pred, [1, -1])
        label = torch.reshape(label, [1, -1])
        loss = loss_calculation(pred, label)
        pred = torch.argmax(pred, dim=1, keepdim=False)
        label = torch.argmax(label, dim=1, keepdim=False)
        for i, j in zip(pred, label):
            if (i == j):
                sum = sum + 1
        Val_y.append(label)
        Val_pred.append(pred)
    print("epoch:{},val_acc:{}".format(epoch + 1, sum / len(val_x)))
    evaluator(Val_y, Val_pred)

###Test#####
Test_pred = []
for x,x1 in zip(test_x,test_x1):
    x = x.to(device)
    x1 = x1.to(device)
    pred = model(x,x1)
    pred = torch.argmax(pred, dim=0, keepdim=False)
    pred = pred.item()
    Test_pred.append(bool(pred))

print(Test_pred)