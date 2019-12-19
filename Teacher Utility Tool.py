import tkinter as tk
from tkinter import messagebox
import csv
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import numpy as np
from PIL import Image,ImageTk

def selection():
    options=['Scaling to Pass','Scaling to Next Grade','Final Grade Prediction','Grace Marks']
    output1=tk.Tk()
    output1.title('Module Selection')
    canvas3=tk.Canvas(output1,width=500, height=270,bg='light blue',relief='raised')
    canvas3.pack()

    P=Image.open("p.png")
    I=ImageTk.PhotoImage(P,master=output1)
    lbl2=tk.Label(output1,image=I)
    lbl2.image=P
    canvas3.create_window(250,100,window=lbl2)
    lbl1=tk.Label(output1,text='Select the option you want to run',bg='light blue')
    lbl1.config(font=('times new roman',15))
    canvas3.create_window(250,170,window=lbl1)
    global var
    var=tk.StringVar(output1)
    var.set('Scaling to Pass')
    
    entry=tk.OptionMenu(output1,var,*options)
    canvas3.create_window(250,210,window=entry)

    openfile=tk.Button(output1,text='Continue',command=name,bg='light blue')
    openfile.config(font=('times new roman',12))
    canvas3.create_window(250,250,window=openfile)
    canvas3.pack()
    output1.mainloop()

def mergesort(lst,start,end,col):
    if start<end:
        mid=(start+end)//2
        mergesort(lst,start,mid,col)
        mergesort(lst,mid+1,end,col)
        merge(lst,start,mid,end,col)
def merge(lst,start,mid,end,col):
    Left=lst[start:mid+1]
    Right=lst[mid+1:end+1]
    x=start
    j=i=0
    a=len(Left)
    b=len(Right)
    while a>0 and b>0:
        c=Left[i][col]
        d=Right[j][col]
        if (c)<(d):
            lst[x]=Left[i]
            x+=1
            i+=1
            a-=1
        elif (c)>(d):
            lst[x]=Right[j]
            j+=1
            x+=1
            b-=1
        else:
            lst[x]=Left[i]
            x+=1
            i+=1
            a-=1
            lst[x]=Right[j]
            j+=1
            x+=1
            b-=1
    while a>0 and b==0:
        lst[x]=Left[i]
        x+=1
        i+=1
        a-=1
    while b>0 and a==0:
        lst[x]=Right[j]
        j+=1
        x+=1
        b-=1
def scaling_fails(lst,minimum):
    dist={}
    for student in lst:
        dist[student[0]]=': This student needs '+str(round(minimum-eval(student[1]),2))+' percent to pass'
    return dist
##    for x in dist:
##        print(x,':',dist[x])
def appointing_grades(Tuple,Grade_dst):
    for student in Tuple:
        for grade in Grade_dst:
            lst=[]
            grade=grade.split('-')
            for num in range(int(grade[0]),int(grade[1])+1):
                lst.append(num)
            if round(eval(student[1])) in lst:
                grade='-'.join(grade)
                student.append(Grade_dst[grade])
def failures(Tuple,minimum):
    failures_lst=[]
    count=0
    for student in Tuple:
        if eval(str(student[1]))<minimum:
            failures_lst.append(student)
            count+=1
        else:
            break
##    print('Total number of failed students:',count)
##    for i in failures_lst:
##        print(i)
    return [failures_lst,count]
def entername():
    try:
        F=open(txt.get()+'.csv','r')
        file=csv.reader(F)
        global lst
        lst=[]
        for i in F:
            i=i.split(',')
            k=i[-1]
            if '\n' in k:
                k=k[:-1]
            s=i[0]
            if '\xa0' in s:
                s=s.split()
                for j in s:
                    if j=='0':
                        s.remove(j)
                        break
                    elif j=='a' or j=='\\' or j=='a':
                        s.remove(j)
                s=' '.join(s)
            lst.append([s,k])
        lst=lst[1:]
        global Tuple
        global Grade_dst
        Grade_dst={
        '96-100':'A+',
        '90-95':'A',
        '85-89':'A-',
        '80-84':'B+',
        '75-79':'B',
        '70-74':'B-',
        '67-69':'C+',
        '64-66':'C',
        '60-63':'C-',
        '0-59':'F',
        }
        Tuple=[]
        output=tk.Tk()
        output.title('Students Final Percentages')
        canvas2=tk.Canvas(output,width=500,height=500,bg='light blue',relief='raised')
        canvas2.pack()
        for line in lst:
            Tuple.append([line[0],line[-1]])
        appointing_grades(Tuple,Grade_dst)
        NewTuple=Tuple.copy()
        dataset=pd.DataFrame(NewTuple,columns=['Name','Percentage','Grade'])
        first=tk.Label(output,text='The students and their final percentages and grades are.')
        canvas2.create_window(250,20,window=first)
        final=tk.Label(output,text='Answer')
        canvas2.create_window(250,250,window=final)
        final["text"]=dataset
        mergesort(Tuple,0,len(Tuple)-1,1)
        global minimum
        minimum=60
        global failures_lst
        failures_lst=failures(Tuple,minimum)
        Select_Program=selection()
    
    except:
        if len(txt.get())==0:
            messagebox.showinfo('An error occured','Please enter a filename.')
        else:
            messagebox.showinfo('An error occured','Oops! This file does not exist in this folder')
def minimum_grace_for_max_pass(lst,minimum,initial_fail):
    dist={}
    number=1
    max_grace=txtnew2.get()
    while number<=int(max_grace):
        lst2=lst.copy()
        for no in range(len(lst2)):
            lst2[no]=[lst2[no][0],eval(lst2[no][1])+number]
        recent_fails=failures(lst2,minimum)
        dist[number]=[initial_fail-int(recent_fails[1])]
        number+=1
##    for i in dist:
##        print('If you award '+str(i)+' mark/s :',str(dist[i][0])+' student/s will pass.')
    return dist
def get_grade(Grade_dst,number):
    for grade in Grade_dst:
            lst=[]
            grade=grade.split('-')
            for num in range(int(grade[0]),int(grade[1])+1):
                lst.append(num)
            if number in lst:
                grade='-'.join(grade)
                return Grade_dst[grade]
def scaling_all(Tuple,Grade_dst2):
    scale_up={}
    for student in Tuple:
        bound=Grade_dst2[student[2]].split('-')
        upperbound=int(bound[1])+1
        scale_up[student[0]]=': This student needs '+str(round(upperbound-eval(student[1]),2))+' percent to change their grade from '+student[2]+' to '+get_grade(Grade_dst,upperbound)
    return scale_up
##    for student in scale_up:
##        print(student,scale_up[student])
def Predict():
    url = "Book1.csv"
    columns = ["Name","Quiz 1 (10)","Quiz 2 (10)","Quiz 3 (10)","Quiz 4 (10)","Quiz 5 (15)","Average Quiz","HW 1 (10)","HW 2 (10)","HW 3 (25)","HW 4 (10)",'HW 5 (10)',"Average HW","Report (15)","Presentation (15)","Final (30)","Total Marks"]
    dataset = pd.read_csv(url, names=columns)
    
    #print(dataset)
    #print(dataset.groupby("Total Marks").size())
    # dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    #plt.show()

    #dataset.hist()
    #plt.show() #Plot of data
    #scatter_matrix(dataset)
    #plt.show()  #Plot of the relations
    array = dataset.values
    X = array[:,1:15]
    Y = array[:,15]
    Y=Y.astype('int')
    # print(array)
    # print(X)
    # print(Y)
    validation_size =0.2
    seed =8
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    seed =13
    # print(X_train)
    # print(Y_train)
    # # print(X_validation)
    # # print(Y_validation)
    # print(len(X_validation))
    # print(len(Y_validation))
    scoring = "accuracy"
    models = []
    models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(gamma='auto')))
    ##evaluate each model in turn
    results = []
    scores=[]
    names = []
    for name, model in models:
      kfold = model_selection.KFold(n_splits=20, random_state=seed)
      cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold,   scoring=scoring)
      results.append(cv_results)
      names.append(name)
      msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
      print(msg)
      scores.append(cv_results.mean())

    maxi=0
    for i in range(len(scores)):
      if scores[maxi]<scores[i]:
        maxi=i
    # fig = plt.figure()
    # fig.suptitle('Algorithm Comparison')
    # ax = fig.add_subplot(111)
    # plt.boxplot(results)
    # ax.set_xticklabels(names)
    # plt.savefig("g.png")

    knn = KNeighborsClassifier()
    lr=LogisticRegression(solver='liblinear', multi_class='ovr')
    lda=LinearDiscriminantAnalysis()
    cart=DecisionTreeClassifier()
    nb=GaussianNB()
    svm=SVC(gamma='auto')

    if maxi==0:
      knn.fit(X_train, Y_train)
      predictions = knn.predict(X_validation)
    elif maxi==1:
      lr.fit(X_train, Y_train)
      predictions = lr.predict(X_validation)
    elif maxi==2:
      lda.fit(X_train, Y_train)
      predictions = lda.predict(X_validation)
    elif maxi==3:
      cart.fit(X_train, Y_train)
      predictions = cart.predict(X_validation)
    elif maxi==4:
      nb.fit(X_train, Y_train)
      predictions = nb.predict(X_validation)
    elif maxi==5:
      svm.fit(X_train, Y_train)
      predictions = svm.predict(X_validation)

    print(accuracy_score(Y_validation, predictions))
    print(confusion_matrix(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

    # model = KNeighborsClassifier()
    # model.fit(X_train, Y_train)
    # # save the model to disk
    # filename = 'finalized_model.sav'
    # joblib.dump(knn, filename)
     
    # # # some time later...
     
    # # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_validation, Y_validation)
    # print(result)
    data_to_predict=pd.read_csv("Book2.csv",names=columns)
    array1=data_to_predict.values
    Name=txtnew.get()
    Names=array1[:,0]
    NameInd=""
    for i in range(len(Names)):
      if Names[i]==Name:
        NameInd=i
        break
    if NameInd=="":
      ynew=["Name Not Found"]
    else:
      Xnew = [array1[i][1:15]]
      if maxi==0:
        ynew = knn.predict(Xnew)
      elif maxi==1:
        ynew = lr.predict(Xnew)
      elif maxi==2:
        ynew = lda.predict(Xnew)
      elif maxi==3:
        ynew = cart.predict(Xnew)
      elif maxi==4:
        ynew = nb.predict(Xnew)
      elif maxi==5:
        ynew = svm.predict(Xnew)

      #print("X=%s, Predicted=%s" % (Xnew[0], ynew[0]))
    Answer=ynew[0]
    if Answer=="Name Not Found":
        messagebox.showinfo('Name not found','Please enter a valid name.')
    else:
        Input=tk.Tk()
        Input.title('Output')
        canvas4=tk.Canvas(Input,width=250, height=150,bg='light blue',relief='raised')
        canvas4.pack()
        L1=tk.Label(Input,text="The predicted final marks are "+str(Answer))
        canvas4.create_window(125,20,window=L1)
def Grace():
    data=minimum_grace_for_max_pass(Tuple,minimum,failures_lst[1])
    dataset=pd.DataFrame.from_dict(data,orient='index',columns=['GraceMarks  StudentsPassed'])
    Inputscaling3=tk.Tk()
    Inputscaling3.title('Output')
    canvas9=tk.Canvas(Inputscaling3,width=500,height=500,bg='light blue',relief='raised')
    L1=tk.Label(Inputscaling3,text='Answer')
    canvas9.create_window(250,250,window=L1)
    L1["text"]=dataset
    canvas9.pack()
def name():
    if (var.get())=='Scaling to Pass':
        data=scaling_fails(failures_lst[0],minimum)
        dataset=pd.DataFrame.from_dict(data,orient='index',columns=['Marks needed'])
        Inputscaling=tk.Tk()
        Inputscaling.title('Output')
        canvas6=tk.Canvas(Inputscaling,width=500,height=500,bg='light blue',relief='raised')
        
        L1=tk.Label(Inputscaling,text='Answer')
        canvas6.create_window(250,250,window=L1)
        L1["text"]=dataset
        canvas6.pack()
    elif (var.get())=='Final Grade Prediction':
##        Predict()
        predictinput = tk.Tk()
        predictinput.title('DSA Project')
        canvas7=tk.Canvas(predictinput,width=250,height=250,bg='light blue',relief='raised')
        canvas7.pack()
        lblnew=tk.Label(predictinput,text="Enter the student's name",bg='light blue')
        lblnew.config(font=('times new roman',15))
        canvas7.create_window(125,50,window=lblnew)
        global txtnew
        txtnew=tk.Entry(predictinput,width=30)
        canvas7.create_window(125,90,window=txtnew)
        global openfile
        openfile=tk.Button(predictinput,text='Continue',command=Predict,bg='light blue')
        openfile.config(font=('times new roman',12))
        canvas7.create_window(125,130,window=openfile)      
    elif (var.get())=='Scaling to Next Grade':
        Grade_dst2={
            'A+':'96-100',
            'A':'90-95',
            'A-':'85-89',
            'B+':'80-84',
            'B':'75-79',
            'B-':'70-74',
            'C+':'66-69',
            'C':'63-66',
            'C-':'60-63',
            'F':'0-59',
            }
        data=scaling_all(Tuple,Grade_dst2)
        dataset=pd.DataFrame.from_dict(data,orient='index',columns=['Marks needed'])
        for i in data:
            print(i,data[i])
        Inputscaling=tk.Tk()
        Inputscaling.title('Output')
        canvas5=tk.Canvas(Inputscaling,width=600,height=600,bg='light blue',relief='raised')
        canvas5.pack()
        L1=tk.Label(Inputscaling,text='Answer')
        canvas5.create_window(300,300,window=L1)
        L1["text"]=dataset
    elif (var.get())=='Grace Marks':
        Inputgrace = tk.Tk()
        Inputgrace.title('DSA Project')
        canvas8=tk.Canvas(Inputgrace,width=250,height=250,bg='light blue',relief='raised')
        canvas8.pack()
        lblnew=tk.Label(Inputgrace,text="Enter the grace marks",bg='light blue')
        lblnew.config(font=('times new roman',15))
        canvas8.create_window(125,50,window=lblnew)
        global txtnew2
        txtnew2=tk.Entry(Inputgrace,width=30)
        canvas8.create_window(125,90,window=txtnew2)
        global openfile2
        openfile2=tk.Button(Inputgrace,text='Continue',command=Grace,bg='light blue')
        openfile2.config(font=('times new roman',12))
        canvas8.create_window(125,130,window=openfile2) 
top = tk.Tk()
top.title('DSA Project')
canvas1 = tk.Canvas(top, width = 500, height = 270, bg = 'light blue', relief = 'raised')
canvas1.pack()
P1=Image.open("p.png")
I1=ImageTk.PhotoImage(P1,master=top)
Ilbl=tk.Label(top,image=I1)
Ilbl.image=P1
canvas1.create_window(250,100,window=Ilbl)
lbl=tk.Label(top,text='Enter the name of the csv file you want to open',bg='light blue')
lbl.config(font=('times new roman',15))
canvas1.create_window(250,170,window=lbl)
txt=tk.Entry(top,width=30)
canvas1.create_window(250,210,window=txt)
openfile=tk.Button(top,text='Open File',command=entername,bg='light blue')
openfile.config(font=('times new roman',12))
canvas1.create_window(220,250,window=openfile)
closefile=tk.Button(top,text='Close',command=top.destroy,bg='light blue')
closefile.config(font=('times new roman',12))
canvas1.create_window(290,250,window=closefile)
top.mainloop()
