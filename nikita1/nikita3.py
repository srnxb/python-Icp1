readinput = input("Enter Python")    # reading input from console

# deleteinput = readinput[0:4]

deleteinput = readinput[0:1] + readinput[3:6]

print(deleteinput)
print("output is "+ deleteinput[::-1])             # reverse the string

read1 = int(input("enter one number"))  # reading input from console
read2 = int(input("enter second number"))  # reading another input from console
print( read1 + read2)