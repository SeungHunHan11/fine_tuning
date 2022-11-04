import torch 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.model_selection import KFold
import sys
import pandas as pd


sys.path.append('finetuning/')

def train(model,train,val,criterion,epochs,lr,device,weight_name,
isbinary=True,progress=True):
    globaliter = 0

    default_dir='runs/resnet'
    os.makedirs(default_dir,exist_ok=True)

    #train_log_dir = 'logs/tensorboard/train'
    #test_log_dir = 'logs/tensorboard/test'

    #train_summary_writer = summary.create_file_writer(train_log_dir)
    #test_summary_writer =  summary.create_file_writer(test_log_dir)

    optimizer=torch.optim.Adam(model.parameters(),lr=lr)

    model.to(device)
    criterion.to(device)
    model.train()

    best_acc=0
    if progress:
        epochs=tqdm(range(epochs))

    acc=[]
    val_acc=[]

    for idx,epoch in enumerate(epochs):

        train_loss=0
        val_loss=0
        correct=0
        val_correct=0

        count=0

        for idx2, (xx,yy) in enumerate(train):

            xx=xx.to(device)
            yy=yy.to(device)
            optimizer.zero_grad()
            output=model(xx)
            
            if isbinary:
                loss=criterion(output,yy.float().unsqueeze(1))
            else:
                loss = criterion(output, yy)

            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            
            if isbinary:
                predicted=torch.round(torch.sigmoid(output))
                correct+=(predicted==yy.unsqueeze(1)).sum().item()

            else:
                _, predicted = torch.max(output.data, 1)
                correct += (predicted == yy).sum().item()
            
            count+=yy.size(0)

            #.as_default():  # 텐서보드에 등록하기
                #summary.scalar('loss', loss.item() , step =globaliter)

        acc.append(round(correct/count,3))

        model.eval()
        
        count=0

        with torch.no_grad():
            for idx3, (xx,yy) in enumerate(val):
                xx=xx.to(device)
                yy=yy.to(device)
                output=model(xx)

                if isbinary:
                    loss=criterion(output,yy.float().unsqueeze(1))
                else:
                    loss = criterion(output, yy)

                val_loss+=loss.item()

                if isbinary:

                    predicted=torch.round(torch.sigmoid(output))
                    val_correct+=(predicted==yy.unsqueeze(1)).sum().item()
                else:
                    _, predicted = torch.max(output.data, 1)
                    val_correct += (predicted == yy).sum().item()

                count+=yy.size(0)

            #with test_summary_writer.as_default():  # 텐서보드에 등록하기
                #summary.scalar('loss', val_loss , step = globaliter)
                #summary.scalar('accuracy', 100 * correct/count , step = globaliter) 

        val_acc.append(round(val_correct/count,3))

        if val_acc[idx]>best_acc:
            best_acc=val_acc[idx]
            torch.save(model.state_dict(),default_dir+'/'+weight_name+'.pt')
            print('Best Trial Renewed')

        prompt='''
        Train Accuracy is {}
        Validation Accuracy is {}
        Current best Accuracy is {}
        '''.format(acc[idx],val_acc[idx],best_acc)

    
        print(prompt)
