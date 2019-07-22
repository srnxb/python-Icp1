# Kate Williams
# Student #39
# Lab 1 Project 2
# Due 6/15/2018

# Accept sentence from user and store it in a variable
theSentence = input("Please enter your sentence: ")

# Output the middle word in the sentence
theSplit = theSentence.split(' ')  # Splits the sentence up into words and stores it in a variable
half = (len(theSplit)/2)  # Variable that refers to the middle of the string
if len(theSplit) % 2 == 0:  # If the number is even
    print("The middle words are "
          + theSplit[int(half - 1)] + ", " + theSplit[int(half)])  # Print the middle words
else:  # If the number is odd
    print("The middle word is " + theSplit[int(half)])  # Print the middle word

# Output the longest word in the sentence
longest = " "  # Blank variable to store the longest word in
word = " "  # Blank variable to store the current word in
for word in theSplit:
    if len(word) > len(longest):  # If the current word is longer than the longest word
        longest = word  # Make the current word the new longest word
print("The longest word is " + longest)  # Print the longest word

# Output the sentence in reverse
print(theSentence[::-1])  # Use the extended slice syntax to print the sentence in reverse