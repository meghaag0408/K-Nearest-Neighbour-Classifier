import random
from matplotlib.pyplot import *
import csv
import math
import operator
from tabulate import tabulate


# Function to load the dataset and create training sample and test sample
# Depending on the split ratio given : Randomsubsampling
def read_file_load_dataset_randomsubsampling(filename, index_of_class, training_sample=[], test_sample=[]):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)
	random.shuffle(dataset) 

	test=[]	
	#Spliting the dataset into training sample : list of lists
	for i in range(0, length/2):
		test = dataset[i].split(',')
		for i in range(0, len(test)):
			if filename=='wisconsin.data':
				if i==0:
					test[i]=0
			if i!=index_of_class:
				if test[i]=='?':
					test[i]=5
				test[i] = float(test[i])

		training_sample.append(test)
		test=[]

	#Spliting the dataset into test sample - with classname(for simplicity): list of lists
	for i in range((length/2), length):
		test = dataset[i].split(',')
		for i in range(0, len(test)):
			if filename=='wisconsin.data':
				if i==0:
					test[i]=0
			if i!=index_of_class:
				if test[i]=='?':
					test[i]=5
				test[i] = float(test[i])
		test_sample.append(test)
		test=[]

	return dataset

# Function to load the dataset and create training sample and test sample
# Depending on the split ratio given : Randomsubsampling
def read_file_load_dataset_fivefoldcrossvalid(begin, end, filename, index_of_class, training_sample=[], test_sample=[]):
	f = open(filename, 'rb')
	dataset = f.readlines()
	length = len(dataset)
	random.shuffle(dataset) 
	test=[]	
	begin_iter_test = int(begin * length)
	end_iter_test = int(end * length)
	begin_iter_training = int(begin * length)
	end_iter_training = int(end * length)

	#Spliting the dataset into training sample : list of lists
	if begin_iter_test==0:
		for i in range(end_iter_test, length):
			test = dataset[i].split(',')
			for i in range(0, len(test)):
				if filename=='wisconsin.data':
					if i==0:
						test[i]=0
				if i!=index_of_class:
					if test[i]=='?':
						test[i]=5
					test[i] = float(test[i])

			training_sample.append(test)
			test=[]
	else:
		for i in range(0, begin_iter_test):
			test = dataset[i].split(',')
			for i in range(0, len(test)):
				if filename=='wisconsin.data':
					if i==0:
						test[i]=0
				if i!=index_of_class:
					if test[i]=='?':
						test[i]=5
					test[i] = float(test[i])

			training_sample.append(test)
			test=[]
		for i in range(end_iter_test, length):
			test = dataset[i].split(',')
			for i in range(0, len(test)):
				if filename=='wisconsin.data':
					if i==0:
						test[i]=0
				if i!=index_of_class:
					if test[i]=='?':
						test[i]=5
					test[i] = float(test[i])

			training_sample.append(test)
			test=[]

	
	#Spliting the dataset into test sample - with classname(for simplicity): list of lists
	for i in range(begin_iter_test, end_iter_test):
		test = dataset[i].split(',')
		for i in range(0, len(test)):
			if filename=='wisconsin.data':
				if i==0:
					test[i]=0
			if i!=index_of_class:
				if test[i]=='?':
					test[i]=5
				test[i] = float(test[i])
		test_sample.append(test)
		test=[]

	return dataset

def calculate_euclidean_distance(variable1, variable2, dimension, index_of_class):
	distance = 0
	for x in range(dimension):
		if x!=index_of_class:
			distance = distance + pow((variable1[x] - variable2[x]), 2)
	return math.sqrt(distance)

def find_k_nearest_neighbours(training_sample, index_of_class, test_instance, k):

	length_training_sample = len(training_sample)
	distances=[]
	for i in range(length_training_sample):
		d = calculate_euclidean_distance(test_instance, training_sample[i], len(test_instance), index_of_class)
		temp = training_sample[i]
		distances.append((temp, d))

	neighbours = []
	distances.sort(key=operator.itemgetter(1))	
	for i in range(k):
		x = distances[i][0]
		neighbours.append(x)   #Appending the first k samples to the neighbours of the given test sample
	return neighbours

#Devise prediction class on the basis of the neighbours obtained
def getprediction(neighbours, index_of_class):
	length_of_neighbours = len(neighbours)
	class_votes_dictionary = {}
	for i in range(length_of_neighbours):
		predicted_class = neighbours[i][index_of_class]
		if predicted_class in class_votes_dictionary:
			x = predicted_class
			class_votes_dictionary[x]+=1
		else:
			x = predicted_class
			class_votes_dictionary[x]=1

	sorted_class_votes=[]
	sorted_class_votes = sorted(class_votes_dictionary.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sorted_class_votes[0][0]


#Calculate the ratio of the total correct predictions out of all predictions made : classification accuracy
def calculate_accuracy(test_verify_sample, predictions, index_of_class):
	correct=0
	length_test_sample = len(test_verify_sample)
	for i in range(length_test_sample):
		if test_verify_sample[i][index_of_class] == predictions[i]:
			correct = correct+1
	accuracy_percentage = (correct/(float(length_test_sample))) * 100
	return accuracy_percentage


#Calculating mean of the accuracy percentage
def calculate_mean(accuracy_percentage_list):
	sum=0
	for i in range(len(accuracy_percentage_list)):
		sum=sum+float(accuracy_percentage_list[i])
	mean = sum/len(accuracy_percentage_list)
	return mean
	

#Calculating standard deviation of the accuracy percentage
def calculate_standard_deviation(accuracy_percentage_list, mean):
	diff_sq = 0
	for i in range(len(accuracy_percentage_list)):
		diff_sq = diff_sq + pow((accuracy_percentage_list[i] - mean), 2)

	diff_sq = diff_sq/len(accuracy_percentage_list)
	return math.sqrt(diff_sq)


def confusion_matrix(filename, predictions, actual_classes):
	#Fetching the name of the classes to dictionary and then to the list
	classes={}
	for i in range(len(actual_classes)):
		if actual_classes[i][(len(actual_classes[i]))-1] == '\n':
				actual_classes[i] = actual_classes[i][0:len(actual_classes[i])-1]
				predictions[i] = predictions[i][0:len(predictions[i])-1]
		if actual_classes[i] in classes:			
			classes[actual_classes[i]]= 1
		else:
			classes[actual_classes[i]]= 1
	c =[]
	for i in classes.keys():
		c.append(i)
	length = len(c)	
	

	#Creating confusion matrix as list -> empty list and hence comparing and increasing the count
	confusion_matrix=[]
	for i in range(length):
		for j in range(length):
			confusion_matrix.append(0)

	count = 0
	for i in range(len(actual_classes)):
		for j in range(length):
			for k in range(length):
				if actual_classes[i] == c[j] and predictions[i] == c[k]:
					count = count +1
					confusion_matrix[j*length+k] = confusion_matrix[j*length+k]+1

	#Printing confusion matrix
	if filename == 'wisconsin.data':
		for i in range(length):
			if c[i] == '2':
				c[i] = 'Benign'
			if c[i]=='4':
				c[i] ='Malignant'
	print "\t\t"+'PREDICTED'
	table = []
	
	#Append Classes name
	L=[]
	L.append('\t')
	L.append('\t')
	for i in range(length):
		L.append(c[i])
	table.append(L)

	#Create Empty Table
	L=[]
	for i in range(length):
		for j in range(length+2):
			if i==length/2:
				if j==0:
					L.append('ACTUAL')
				elif j==1:
					L.append(c[i])
				else:
					L.append('\t')
			else:
				if j==1:
					L.append(c[i])
				else:
					L.append('\t')
		table.append(L)
		L=[]

	#Populate value to the confusion matrix/empty table
	value_index=0
	for i in range(1, length+1):
		for j in range(2, length+2):
			table[i][j] = confusion_matrix[value_index]
			value_index+=1

	print tabulate(table, tablefmt="grid")




if __name__ == '__main__':

	#Taking inputs
	filename = raw_input('Enter the source dataset filename: ')
	index_of_class = input('Enter the index of the class in the given dataset: ')
	k = input('Enter the value of k: ')
	print "Enter 1...........Random Sub Sampling"
	print "Enter 2...........Five Fold Cross Validation"
	choice = input("Enter Choice\n")
	
	if choice == 1:
		accuracy_percentage_list=[]
		for x in range(10):
			#Initiliasing Variables
			training_sample = []
			test_sample=[]
			dataset=[]
			dataset = read_file_load_dataset_randomsubsampling(filename, index_of_class, training_sample, test_sample)
			predictions = []
			for i in range(len(test_sample)):
				neighbours = find_k_nearest_neighbours(training_sample, index_of_class, test_sample[i], k)		
				predicted_output = getprediction(neighbours, index_of_class)
				predictions.append(predicted_output)

		
			#print predictions
			actual_classes = []
			for i in range(len(test_sample)):
				actual_classes.append(test_sample[i][index_of_class])

			accuracy_percentage = calculate_accuracy(test_sample, predictions, index_of_class)
			print '\n'
			print 'ITERATION NO : ' + repr(x+1)
			print 'Accuracy: ' + repr(accuracy_percentage)
			accuracy_percentage_list.append(accuracy_percentage)
			confusion_matrix(filename, predictions, actual_classes)


		print accuracy_percentage_list
		mean = calculate_mean(accuracy_percentage_list)
		sd = calculate_standard_deviation(accuracy_percentage_list, mean)
		print 'Mean: ' + repr(mean)
		print 'Standard Deviation: ' + repr(sd)

	elif choice == 2:	
		mean_list = []			
		for x in range(10):
			accuracy_percentage_list=[]
			print '\n'
			print 'ITERATION NO : ' + repr(x+1)
			split_ratio = 0.0
			for y in range(5):
				#Initiliasing Variables
				training_sample = []
				test_sample=[]
				dataset=[]
				dataset = read_file_load_dataset_fivefoldcrossvalid(split_ratio, split_ratio+0.2, filename, index_of_class, training_sample, test_sample)
				split_ratio+=0.2
				predictions = []
				for i in range(len(test_sample)):
					neighbours = find_k_nearest_neighbours(training_sample, index_of_class, test_sample[i], k)		
					predicted_output = getprediction(neighbours, index_of_class)
					predictions.append(predicted_output)

		
				#print predictions
				actual_classes = []
				for i in range(len(test_sample)):
					actual_classes.append(test_sample[i][index_of_class])

				accuracy_percentage = calculate_accuracy(test_sample, predictions, index_of_class)
				print 'FOLD #' + repr(y+1)
				print 'Accuracy: ' + repr(accuracy_percentage)
				accuracy_percentage_list.append(accuracy_percentage)
				


			mean = calculate_mean(accuracy_percentage_list)
			sd = calculate_standard_deviation(accuracy_percentage_list, mean)
			mean_list.append(mean)
			print '---------------------------------------------'
			print 'Mean: ' + repr(mean)
			print 'Standard Deviation: ' + repr(sd)
			print '---------------------------------------------'


		
		print 
		print '=================================================='
		grand_mean = calculate_mean(mean_list)
		grand_sd = calculate_standard_deviation(mean_list, grand_mean)
		print 'Grand Mean: ' + repr(grand_mean)
		print 'Grand Standard Deviation: ' + repr(grand_sd)
		print '=================================================='
		confusion_matrix(filename, predictions, actual_classes)
		
	else:
		print "Wrong Choice Entered!"
	

			






