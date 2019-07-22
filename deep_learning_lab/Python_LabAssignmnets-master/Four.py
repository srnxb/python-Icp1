# Kate Williams
# Student #39
# Lab 1 Project 4
# Due 6/15/2018

# Initialize lists of students from both classes
pyStudents = ["Tony Stark", "Bruce Banner", "Natasha Romanov", "Bruce Wayne", "Diana Prince"]
appStudents = ["Clark Kent", "Bruce Wayne", "Diana Prince", "Natasha Romanov"]

pySet = set(pyStudents)  # Make the lists sets first, to make them easier to compare
appSet = set(appStudents)

# Find the students who are attending both classes
print(pySet & appSet)
print(" are the students attending both classes")

# Find the students who are not attending both classes
print(pySet ^ appSet)
print(" are the students not attending both classes")
