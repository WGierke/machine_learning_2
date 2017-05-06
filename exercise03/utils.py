import numpy,matplotlib
import matplotlib.pyplot as plt

# Extract a dataset of images patches from a collection of images.
#
# input:  None
# output: an array of size (#patches x 432) containing the image patches
#
def load():
	def loadimage(name):
		s = 12
		Z = plt.imread("images/%s.jpg"%name)
		Z = numpy.array(Z)
		X = [[Z[s*i:s*i+s,s*j:s*j+s,:] for j in range(Z.shape[1]/s)] for i in range(Z.shape[0]/s)]
		return numpy.array(X).reshape([-1,s*s*3])*1.0

	X = []
	for name in ['car','paper','plane','restaurant','stairs','tulips','pens','nature']:
		X += [loadimage(name)]
	X = numpy.concatenate(X,axis=0)
	X = X - X.mean(axis=0)
	X = X / X.std()
	return X[numpy.random.permutation(len(X))]


# Scatter plot
#
# input: the vector of x- and y-coordinates and the axis labels
# output: None
#
def scatterplot(x,y,xlabel='',ylabel=''):
    assert(x.ndim==1 and y.ndim==1 and len(x)==len(y))
    matplotlib.pyplot.xlabel(xlabel)
    matplotlib.pyplot.ylabel(ylabel)
    matplotlib.pyplot.plot(x,y,'.',ms=2)
    matplotlib.pyplot.show()


# Render the input matrix as a collection of image patches
#
# input:  an array of size (#patches x 432)
# output: None
#
def render(x):
    
	assert(x.ndim == 2)

	x = x[:(len(x)/25)*25] # make the number of rows a multiple of 25 for visualization purposes

	s = 12

	x = x.reshape([x.shape[0],s,s,3])

	# normalize (and map nonlinearly to make images look nice on the screen)
	x = x - x.mean()
	x = x / x.std()
	x = numpy.tanh(0.5*x)*0.5+0.5

	# create the mosaic
	h,w = x.shape[0]/25,25
	x = x.reshape([h,w,s,s,x.shape[3]])
	z = numpy.ones([h,w,s+2,s+2,x.shape[4]])
	z[:,:,1:-1,1:-1,:] = x
	z = z.transpose([0,2,1,3,4]).reshape([h*(s+2),w*(s+2),3])

	# display the mosaic
	plt.figure(figsize=(10,10))
	plt.imshow(z)
	plt.axis('off')
	plt.show()

