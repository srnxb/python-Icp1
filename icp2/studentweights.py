
n = int(input("Enter number of elements : "))
list1 = []
list2 = []
for x in range(n):
    list1.append(float(input("Enter "+str(x+1)+" element : ")))
    list2 = (i*0.454 for i in list1)
dictionary = dict(zip(list1, list2))
print(dictionary)
