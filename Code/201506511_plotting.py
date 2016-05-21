from matplotlib.pyplot import *

def data():
    f = open('iris.data', 'rb')
    dataset = f.readlines()
    length = len(dataset)
    test=[]
    lists=[]
    for i in range(0, length):
        test = dataset[i].split(',')
        lists.append(test)
        test=[]

    return [map(float, l[:4]) for l in lists], [l[-1] for l in lists]

index=0
preindex=0
i=1.5
j=-0.5
#Obtaining the numerical values and the class labels:
matrix, labels = data()
xcord1 = []
ycord1 = []
# sepal width
xcord2 = []
ycord2 = []
 # petal width
xcord3 = []
ycord3 = []
decision_line_x=[]
decision_line_y=[]
y = 3
x = 1 
#Locating x and y coordinates of the points given ( sepal width and petal width)
for n, elem in enumerate(matrix):
    if labels[n] == 'Iris-setosa\n':
        length=len(xcord1)
        xcord1.insert(length,matrix[n][x])
        length=len(ycord1)
        ycord1.insert(length,matrix[n][y])
    if labels[n] == 'Iris-versicolor\n':
        length=len(xcord2)
        xcord2.insert(length,matrix[n][x])
        length=len(ycord2)
        ycord2.insert(length,matrix[n][y])
    if labels[n] == 'Iris-virginica\n':
        length=len(xcord3)
        xcord3.insert(length,matrix[n][x])
        length=len(ycord3)
        ycord3.insert(length,matrix[n][y])


#Plotting the decision bounday
while i<5:
    j=0
    while j<3:
        max_value=10000000
        temp1='iris'
        preindex=index        
        for k in range(150):
            t = matrix[k][1]-i
            t2 = matrix[k][3]-j
            temp = pow(t,2)+pow(t2,2)
            if max_value>temp:
                max_value=temp
                if labels[k]=='Iris-setosa\n':
                    index=1
                elif labels[k]=='Iris-versicolor\n':
                    index=2
                else :
                    index=3
        if preindex==index:
            pass
        if preindex !=index:
            decision_line_x.append(i)
            decision_line_y.append(j)
        j=j+0.1
    i=i+0.1

#Plotting the graph
ax = figure().add_subplot(111)
type4 = ax.scatter(decision_line_x, decision_line_y, 10, color ='grey')

ax.set_title('2-D Plot for Iris dataset', fontsize=14)
ax.legend([ax.scatter(xcord1, ycord1, s=40, c='red'), ax.scatter(xcord2, ycord2, s=40, c='green'), ax.scatter(xcord3, ycord3, s=40, c='blue')], ["Iris Setosa", "Iris Versicolor", "Iris Virginica"], loc=2)
ax.set_xlabel('Sepal Width (cm)')
ax.set_ylabel('Petal width (cm)')
ax.grid(True,linestyle='-',color='0.75')

show()