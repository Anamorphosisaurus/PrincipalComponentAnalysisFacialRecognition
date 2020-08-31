import cv2
import os
import numpy as np
import sys
import numpy
#numpy.set_printoptions(threshold=sys.maxsize)
##Getting the training images in vector form##
Training_Image_List = os.listdir('C:/Program Files/AnamorphosisaurusPCA/training')                          #saving directory
Number_Of_Training_Images = len(Training_Image_List)                                                        #getting the number of training images
Size_Image = cv2.imread('C:/Program Files/AnamorphosisaurusPCA/training/1.jpg',0)                           #using opencv to read the size of each image in the data set 
[Height,Length] = np.shape(Size_Image)                                                                      #using the first image in the data set to get the height and length of each image
Vector_Length = Height*Length                                                                               #multiplying height and length to get the size of the vectors
X = np.zeros((Vector_Length,Number_Of_Training_Images))                                                     #creating a zero matrix for the training image vectors


for i in range (0,Number_Of_Training_Images):                                                               #starting a for loop to collect all vectors
    #Training_Number = 0                                                                                     
    Training_Image_Path = 'C:/Program Files/AnamorphosisaurusPCA/training/'                                 #saving the path to the training images
    Training_Number = i                                                                                     #selected training image
    Training_Number_String = str(Training_Number)                                                           #change int to string                                                       
    Training_Image_Type = '.jpg'                                                                            #file type
    Training_Image_Location = Training_Image_Path + Training_Number_String + Training_Image_Type            #putting the path, training image number and file type together
    Training_Image_Gray = cv2.imread(Training_Image_Location,0)                                             #making image gray scale
    for j in range (0,Length):                                                                              #creating for loop to create vector from a matrix
        X[j*Length:j*Length+Length,i] = Training_Image_Gray[j,:]                                            #creating the vectors

##Getting the mean vector##   
Mean_Vector = np.zeros((Vector_Length,1))                                                                   #creating a zero matrix for the mean vector
for i in range(0,Vector_Length):                                                                            #for loop to put the mean into each vector slot which creates a mean vector
    Mean_Vector[i,0]= (sum(X[i,:])/Number_Of_Training_Images)                                               #adding ith element of each vector and dividing each of them by te total number of training images
##Getting the S vector##
#print(Mean_Vector)
    
S = np.zeros((Vector_Length,Number_Of_Training_Images))                                                     #making a zero matrix for the Scatter matrix
for i in range(0,Number_Of_Training_Images):                                                                #for loop to create Scatter matrix
    S[:,i]=X[:,i]-Mean_Vector[:,0]                                                                          #subtracting the mean vector from the training image vectors to get the Scatter
#print(S)

##Covariance_Matrix##
S_Transpose = S.transpose()                                                                                 #getting the transpose of the Scatter vector.
Covariance_Matrix = np.dot(S_Transpose,S)                                                                   #multiplying Scatter transpose and the scatter matrix
#print(Covariance_Matrix)

Eigenvalues, Eigenvectors = np.linalg.eig(Covariance_Matrix)                                                #using np to get the eigenvalues and eigenvectors
Inverse_Eigen_Vectors = np.linalg.inv(Eigenvectors)                                                         #getting the inverse eigen vectors
diagonal = Inverse_Eigen_Vectors.dot(Covariance_Matrix).dot(Eigenvectors)                                   #getting the diagonal matrix
#print(diagonal.round(5))

Count = 0                                                                                                   #setting count to 0
for i in range(0,Number_Of_Training_Images):                                                                #for loop to create the diagonal matrix
    if(diagonal[i,i] > 1):                                                                                  #if diagonal elements are greater than 0, 
        Count = Count + 1                                                                                   #move to next count


Covariance = np.zeros((Number_Of_Training_Images,Count))                                                    #building the covarience
for i in range(0,Count):
    if(diagonal[i,i] > 1):
        Covariance[:,i] = Eigenvectors[:,i]


Eigenfaces = np.dot(S,Covariance)                                                                           #getting the eigenfaces
[w,e] = np.shape(Eigenfaces)                                                                                #getting the shape of eigenfaces



Projection = np.zeros((Count,Count))                                                                        #building the zero matrix for the projection
for i in range(0,Count):                                                                                    
    Image_Vector = np.dot(Eigenfaces.transpose(),S[:,i])
    Projection[:,i] = Image_Vector                                                                          #collecting the projection
#print(Projection)
         
T_X = np.zeros((Vector_Length,1))                                                                           #creating a zero vector for the training image
min_dist = np.zeros((900,2))
M = cv2.imread('C:/Program Files/AnamorphosisaurusPCA/test/10.jpg',0)                                       #getting the shap of the images                                       
[L,H] = np.shape(M)                                                                                         #getting the L and H of the image
X = L - 100                                                                                                 #setting up the x for the crawler
Y = H - 100                                                                                                 #setting up the y  for the crawler

frame = 1
f = 0

for x in range (0,Y,10):                                                                                    #crawler for loop
    for i in range(0,X,10):                                                                                 #crawler for loop
        Test_Image_Gray = (M[x:x+100,i:i+100])                                                              #getting 100 by 100 pixel segments.
        path = 'C:/Program Files/AnamorphosisaurusPCA/crawl/'                                               #going to the crawler directory
        num = str(frame)                                                                                    #int to string                                                                               
        finalPath = path+num+'.jpg'                                                                         #type of file
        cv2.imwrite(finalPath,Test_Image_Gray)                                                              #writing each segment to the path
        frame = frame + 1                                                                                   #next frame
        

        for j in range (0,Length):                                                                          #making it a vector
                T_X[j*Length:j*Length+Length,0] = Test_Image_Gray[j,:]

        Test_Mean_Vector = T_X - Mean_Vector                                                                # getting the Scatter of the text image

        Test_Image_Projection = np.dot(Eigenfaces.transpose(),Test_Mean_Vector)                             #getting the projection of the test image
        #print(Test_Image_Projection)

        distance = np.zeros((1,Count))                                                                      
        v = np.zeros((1,Count))
        for i in range (0,Count):
            v[0,:] = Test_Image_Projection[:,0]-Projection[:,i]                                             #getting the difference between the training and test projection
            distance[0,i] = pow(np.linalg.norm(v),2)                                                        #equation for the distance
    


        min_dist[f,0] = distance.min()                                                                      #collecting the minimum distance for each segment
        min_dist[f,1] = f                                                                                   #saving the place
        f = f + 1
N = 0
res = [min(i) for i in zip(*min_dist)][N]

print(res)
for i in range (0,900):                                                                                     #printing the minimum 
    if (min_dist[i,0] == res):
        rec = i
        print(i)

path = 'C:/Program Files/AnamorphosisaurusPCA/crawl/'
num = str(rec)
finalPath = path+num+'.jpg'
New_Image = cv2.imread(finalPath,0)



path = 'C:/Program Files/AnamorphosisaurusPCA/training/'
num = str(27)
finalPath = path+num+'.jpg'
cv2.imwrite(finalPath,New_Image)

        








    
    



    
 
