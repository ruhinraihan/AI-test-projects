from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import  random

style.use('ggplot')

xs = [1, 2, 3, 4, 5]
ys = [5, 4, 6, 5, 6]

#plt.scatter(xs, ys)
#plt.show()
xs = np.array(xs, dtype=np.float64)
ys = np.array(ys, dtype=np.float64)

def create_dataset(totalPoints, variance, step=2, correlation=False):
	st = 1
	ys = []
	xs = []
	#print(correlation)
	for i in range(totalPoints):
		st += random.randrange(-variance, variance)
		ys.append(st)
		xs.append(i)
		if correlation and correlation == 'pos':
			st += step
		elif correlation and correlation == 'neg':
			st -= step

	return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

def best_fit_slop_intercept(xs, ys):
	m = (mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
	b = mean(ys) - m*mean(xs)
	return m, b

def square_error(ys_origin, ys_line):
	return sum((ys_line-ys_origin)**2)

def coefficient_of_determination(ys_origin, ys_line):
	y_mean_line = [mean(ys_origin) for _ in ys_origin]
	#print(y_mean_line)
	square_error_regr = square_error(ys_origin, ys_line)
	square_error_y_mean = square_error(ys_origin, y_mean_line)
	return 1 - (square_error_regr / square_error_y_mean)

xs, ys = create_dataset(40, 10, 2, 'pos')
m, b = best_fit_slop_intercept(xs, ys)
#print(m, b)

regression_line = [(m*x)+b for x in xs]

#print(regression_line)
# x_predict = 7
# y_predict = m*x_predict+b
#
r_square = coefficient_of_determination(ys, regression_line)
print(r_square)

plt.scatter(xs, ys, label='train_data')
# plt.scatter(x_predict, y_predict, color='g', label='predict')
plt.plot(xs, regression_line, label='regression line')
plt.legend(loc=4)
plt.show()


