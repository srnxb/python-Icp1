import numpy
num = numpy.random.randint(1, high=20, size=15, dtype='int');
print('Random array generated',num)
#num = numpy.amax(num) = 0
print ('Replacing Max No. by 0',numpy.where(num==numpy.amax(num), 0, num))
