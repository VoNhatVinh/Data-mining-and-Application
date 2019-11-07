#import library
import csv
import numpy as np
import random
import sys
from scipy.spatial.distance import cdist
from copy import deepcopy


#------------------------------------------------------------------------------------
'''Ham doc file csv: 
input: file csv
output:
attribute: dong chua ten thuoc tinh
data: cac dong chua gia tri thuoc tinh
'''
def read_CSV(file_name):   
    
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        file_read = []
        for row in csv_reader:
            file_read.append(",".join(row))

        # processing attribute_name
        attribute = file_read[0]  # First line is name of N attributes
        attribute = attribute.split(',')

        # processing data
        data = file_read[1:]  # data in another lines
        data = [line.split(',') for line in data]
        data = [list(map(float, i)) for i in data]  # convert string to int
        data = np.asarray(data)  # convert list to array

    return data, attribute

#-------------------------------------------------------------------------------
'''
input:  giá trị thuộc tính (Data), cụm  mà data thuộc về(cluster), các tâm cụm (center)
output: độ lỗi SSE
'''
def SSE(data, cluster, centroid):
    sum_err = 0
    x = data.shape[0]
    for i in range(x):
        sum_err += np.sum((centroid[cluster[i]] - data[i])**2)
    return sum_err

#------------------------------------------------------------------------------------
'''
input: giá trị các thuộc tính (data), Các tâm (centers)
output: cụm mà điểm data thuộc về 
'''
def get_clusters(data, centroids):
   
    # Tinh khoang cach giua cac diem data va center
    distance = cdist(data, centroids)
    
    # Tra ve cụm có center gan nhat voi data
    cluster = np.argmin(distance, axis=1)
    
    return cluster

#--------------------------------------------------------------------------------------
'''
input: giá trị các thuộc tính (data), số cụm (k), các cụm chứa data (clusters)
output: tâm mới sau khi tính trung bình các điểm trong 1 cụm
'''
def update_centroids(data, clusters, K):
    
    centers = np.zeros((K, data.shape[1]))
    
    for k in range(K):
        # cac diem thuoc ve cluster thu k
        data_k = data[clusters == k, :]
        # tinh trung binh cum
        centers[k, :] = np.mean(data_k, axis=0)
    
    return centers

#--------------------------------------------------------------------------------
'''
input: số cụm (k), giá trị thuộc tính (data)
output: các tâm cụm sau khi khởi tạo ngẫu nhiên từ data
'''
def init_k_centroids(k, data):
    
    print('Init starting points:\n')

    #Khoi tao ngau nhien k centroid
    idx = np.random.randint(len(data), size=k) #khoi tao ngau nhien chi so
    centroid = data[idx, :] #lay tam trong data bang chi so idx
    centroid = np.asarray(centroid)

    for i in range(k):
        print("Cluster", i, ": ", centroid[:, i])
   
    return centroid

#--------------------------------------------------------------------------------
'''
input: tam cum moi (new_centroids) và tam cum cu (centroid)
output: true neu 2 tam khong co thay doi
'''
def centroids_equal(old_centroids, new_centroids):
    return (set([tuple(a) for a in old_centroids]) == set([tuple(a) for a in new_centroids]))

#--------------------------------------------------------------------------------
'''
input: giá trị thuộc tính (data), số cụm (k)
output: các tâm (centers), cụm (cluster)
'''
def kmeans(data, k):
    print("K-mean Running")

    #Khoi tao cum
    centers = [init_k_centroids(k, data)]

    clusters = []
    iters = 0 # so vong lap    
    while True:
        
        print("Iteration ",iters+1,":")
        #cluster ma data thuoc ve
        clusters.append(get_clusters(data, centers[-1]))

        #Truong hop cum khong co phan tư, ta khoi tao lai centroid
        for i in range(k):
            if (sum(clusters[-1] == i) == 0):
                centers = [init_k_centroids(k, data)]
                clusters.append(get_clusters(data, centers[-1]))

        #tam moi sau khi da tinh trung binh
        new_centers = update_centroids(data, clusters[-1], k)

        sse = SSE(data, clusters[-1], new_centers)
        print("SSE = %f\n" % sse)

        #neu tam moi va tam cu khong thay doi thi ngung thuat toan
        if centroids_equal(centers[-1], new_centers):
            break

        centers.append(new_centers)
        iters += 1
        
    
    print("Done K-mean")
    return (centers, clusters)

#--------------------------------------------------------------------------------
'''
input: tên thuộc tính (attribute), giá trị thuộc tính (Data), 
    cụm data thuộc về (cluster_data), ten file xuat ra (fo)
output: in ra tên thuộc tính, giá trị thuộc tính và cụm mà data thuộc về
'''
def write_asgn(attributes, data, cluster_data, fo="assignments.csv"):
    print("Writing to assignment.csv")
   
    with open(fo, mode='w') as csv_file:
        
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quoting=csv.QUOTE_NONE,  lineterminator='\n')
        attributes.append("cluster")
        csv_writer.writerow(attributes)
        h = data.shape[0]
        newData = data
        
        for i in range(h):
            newData = np.append(data[i], cluster_data[i])
            newData[-1] = int(newData[-1])
            csv_writer.writerow(newData)

#-------------------------------------------------------------------------------
'''
input:  số cụm (k), giá trị thuộc tính (data), tâm cụm (centroid), tên thuộc tính (attribute),
cụm  mà data thuộc về(cluster), ten file xuat ra (fo)
output: theo yêu cầu đề bài
'''
def write_model(k, data, centroid, attribute, cluster, fo="model.txt"):
    SSE_err = SSE(data, cluster, centroid)  # compute sum of squared error
    print("Writing to model.txt")
    
    with open(fo, mode='w') as txt:
        txt.write("Within cluster sum of squared errors: %f\n" % SSE_err)
        txt.write("Cluster centroids:")
        txt.write("\t\t\t\t Cluster# \n")
        txt.write("Attribute \t")
        for i in range(k):
            txt.write("%d\t\t" % i)
        txt.write('\n')
        txt.write("\t\t")
        for i in range(k):
            txt.write("(%d)\t\t" % (sum(cluster == i)))
        txt.write('\n')
        txt.write(
            "========================================================================================================================================\n")
        for i in range(len(attribute)):
            # for "Products" and "Purchase" because the word's length is over one tab
            if(i == 1 or i == (len(attribute) - 1)):
                txt.write("%s\t" % attribute[i])
            else:
                txt.write("%s\t\t" % attribute[i])
            for j in range(k):
                txt.write("%f\t" % round(centroid[j][i], 4))
            txt.write("\n")


#-------------------------------------------------------------------------------
def main():
    Input = sys.argv[1]  # input type csv
    Output_model = sys.argv[2]  # output model
    Output_asgn = sys.argv[3]  # output assigments
    k = int(sys.argv[4])  # K clusters in Kmean
    
    # read file and save data, title
    data, attributes = read_CSV(Input)
   
    # Kmean process
    centroids, cluster = kmeans(data, k)
    
    # #write model
    write_model(k, data, centroids[-1], attributes, cluster[-1],Output_model)
    
    # write assignments
    write_asgn(attributes, data, cluster[-1],Output_asgn)
    
    print("Done")


if __name__ == '__main__':
    main()