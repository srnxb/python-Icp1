list1 = [1, 3, 6, 2, -1, 2, 8, -2, 9]
result = []
list1.sort()
end=len(list1)-1
for i in range(len(list1)-2):
    start = i + 1  
    while (start < end):
        total = list1[i] + list1[start] + list1[end]
        if (total < 0):
            start = start + 1
        if (total > 0):
            end = end - 1
        if (total == 0):  # 0 is False in a boolean context
            result.append([list1[i],list1[start],list1[end]])
            start = start + 1  # increment l when we find a combination that works
print(result)

