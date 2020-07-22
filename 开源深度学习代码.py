from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark import SparkConf

#if __name__ == '__main__':
sparkConf = SparkConf().setAppName("example_pytorch").setMaster('local[1]')
spark = SparkSession.builder.config(conf=sparkConf).getOrCreate()

from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import Row
from scipy.stats import mode



'''db_name='prj28'
df1_p=spark.sql("select * from %s.df_model_input_sample" %db_name)
df1=df1_p.toPandas()'''
import pandas as pd
data=pd.read_csv("/home/cdsw/liuxy30/DF_MODEL_INPUT_SAMPLE.csv").set_index('Cust_Id')
data=data.fillna(0)

#data=data[0:10000]


for i in data.columns:
  if data[i].dtype != 'float64':
    print(i,data[i].dtype)
    
    
fea_notF  = ['df_dt','core_cust_lev_cd','indus_cd','Rating','City_Cd']  
data.drop(data[fea_notF],axis=1, inplace=True)    
  
#=========================
#2.数据预处理=====================================================================
#2.1删除样本缺失值过多的数据

#2.3同值处理
#2.3.1同值性特征识别与处理
equi_fea = []
for i in data.columns:
    try:
        mode_value = mode(data[i][data[i].notnull()])[0][0]
        mode_rate = mode(data[i][data[i].notnull()])[1][0] / data.shape[0]
        if mode_rate > 0.6:
            equi_fea.append([i, mode_value, mode_rate])
    except Exception as e:
        print(i, e)
e = pd.DataFrame(equi_fea, columns=['col_name', 'mode_value', 'mode_rate'])
e.sort_values(by='mode_rate')
#2.3.2处理同一性数据

#mode_rate :取值0.5－0.6之间的特征
#s2 = ['gender_cd',
# 'SIGN_WXYH',
# 'IS_APPYH',
# 'IS_ZFJS',
# 'r6m_atm_channel_setl_amt',
# 'r6m_atm_channel_setl_cnt',
# 'r6m_atm_channel_amt_rate',
# 'r6m_atm_channel_cnt_rate',
# 'R6M_Debit_Crd_Csm_Amt',
# 'R6M_Debit_Crd_Csm_Cnt',
# 'R6M_Debit_Crd_Csm_Ave_Amt',
# 'Max_R6M_Debit_Crd_Csm_Amt']

same_val_fea_to_drop = list(e.col_name.values)

for i in same_val_fea_to_drop:
  if i == 'if_lc':
    print(i)
    same_val_fea_to_drop.remove('if_lc')

data.drop(same_val_fea_to_drop, axis=1, inplace=True)
    
data[['if_lc']]=data[['if_lc']].astype('double')
  

#2.4 筛选变量
#2.4.1特征重要性筛选变量
#2.4.2 IV值和WOE值
#连续型特征计算IV值  
  
#==========================＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝    
    

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import gc
import time

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from sklearn.preprocessing import LabelEncoder, StandardScaler



data_feature = data[data.columns[1:]]
data_label = data[data.columns[0]]

from sklearn.model_selection import train_test_split
train_X,test_X,train_y,y_test = train_test_split(data_feature,data_label,test_size=0.3)

epochs = 30                  
batch_size = 1024
classes = 1
learning_rate = 0.01

### normalized
scaler = StandardScaler()
train_X = scaler.fit_transform(train_X)
test_X = scaler.transform(test_X)


import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import time


class MLP(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout=0.5):
        super(MLP, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)

        self.hidden_1 = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.bn1 = torch.nn.BatchNorm1d(n_hidden)

        self.hidden_2 = torch.nn.Linear(n_hidden, n_hidden//2)
        self.bn2 = torch.nn.BatchNorm1d(n_hidden//2)

        self.hidden_3 = torch.nn.Linear(n_hidden//2, n_hidden//4)  # hidden layer
        self.bn3 = torch.nn.BatchNorm1d(n_hidden//4)

        self.hidden_4 = torch.nn.Linear(n_hidden // 4, n_hidden // 8)  # hidden layer
        self.bn4 = torch.nn.BatchNorm1d(n_hidden // 8)

        self.out = torch.nn.Linear(n_hidden//8, n_output)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden_1(x))  # activation function for hidden layer
        x = self.dropout(self.bn1(x))
        x = F.relu(self.hidden_2(x))  # activation function for hidden layer
        x = self.dropout(self.bn2(x))
        x = F.relu(self.hidden_3(x))  # activation function for hidden layer
        x = self.dropout(self.bn3(x))
        x = F.relu(self.hidden_4(x))  # activation function for hidden layer
        x = self.dropout(self.bn4(x))
        x = self.out(x)
        return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


folds = KFold(n_splits=5, shuffle=True, random_state=2019)
NN_predictions = np.zeros((test_X.shape[0], ))
oof_preds = np.zeros((train_X.shape[0], ))

x_test = np.array(test_X)
x_test = torch.tensor(x_test, dtype=torch.float)
if torch.cuda.is_available():
    x_test = x_test.cuda()
test = TensorDataset(x_test)
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

avg_losses_f = []
avg_val_losses_f = []

for fold_, (trn_, val_) in enumerate(folds.split(train_X)):
    print("fold {}".format(fold_ + 1))

    x_train = Variable(torch.Tensor(train_X[trn_.astype(int)]))
    y_train = Variable(torch.Tensor(train_y[trn_.astype(int), np.newaxis]))

    x_valid = Variable(torch.Tensor(train_X[val_.astype(int)]))
    y_valid = Variable(torch.Tensor(train_y[val_.astype(int), np.newaxis]))

    model = MLP(x_train.shape[1], 512, classes, dropout=0.4)

    if torch.cuda.is_available():
        x_train, y_train = x_train.cuda(), y_train.cuda()
        x_valid, y_valid = x_valid.cuda(), y_valid.cuda()
        model = model.cuda()

    # 二分类
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # Combined with the sigmoid

    train = TensorDataset(x_train, y_train)
    valid = TensorDataset(x_valid, y_valid)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        avg_loss = 0.
        for i, (x_batch, y_batch) in enumerate(train_loader):
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()       # clear gradients for next train
            loss.backward()             # -> accumulates the gradient (by addition) for each parameter
            optimizer.step()            # -> update weights and biases
            avg_loss += loss.item() / len(train_loader)
            # avg_auc += round(roc_auc_score(y_batch.cpu(),y_pred.detach().cpu()),4) / len(train_loader)
        model.eval()

        valid_preds_fold = np.zeros((x_valid.size(0)))
        test_preds_fold = np.zeros((len(test_X)))

        avg_val_loss = 0.
        # avg_val_auc = 0.
        for i, (x_batch, y_batch) in enumerate(valid_loader):
            y_pred = model(x_batch).detach()

            # avg_val_auc += round(roc_auc_score(y_batch.cpu(),sigmoid(y_pred.cpu().numpy())[:, 0]),4) / len(valid_loader)
            avg_val_loss += loss_fn(y_pred, y_batch).item() / len(valid_loader)
            valid_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, epochs, avg_loss, avg_val_loss, elapsed_time))

    avg_losses_f.append(avg_loss)
    avg_val_losses_f.append(avg_val_loss)

    for i, (x_batch,) in enumerate(test_loader):
        y_pred = model(x_batch).detach()

        test_preds_fold[i * batch_size:(i + 1) * batch_size] = sigmoid(y_pred.cpu().numpy())[:, 0]

    oof_preds[val_] = valid_preds_fold
    NN_predictions += test_preds_fold / folds.n_splits

    
for i in oof_preds:
  if i>0.5:
    print()
 

auc = round(roc_auc_score(train_y, oof_preds), 4)
print('All \t loss={:.4f} \t val_loss={:.4f} \t auc={:.4f}'.format(np.average(avg_losses_f), np.average(avg_val_losses_f), auc))

threshold = 0.5
result = []
for pred in NN_predictions:
    result.append(1 if pred > threshold else 0)
    
threshold = 0.5
result = []
for pred in oof_preds:
    result.append(1 if pred > threshold else 0)

from sklearn.metrics import accuracy_score
acc = accuracy_score(train_y, result)   
acc















