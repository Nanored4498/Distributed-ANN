import pylab as pl

eps, ts = [], []
while True:
	line = input().split()
	if line[0] == "END": break
	ep, t = map(float, line)
	eps.append(ep)
	ts.append(t)

pl.plot(eps, ts)
pl.xlabel("Probability of error when sending a message")
pl.ylabel("Time of execution of some feedforward and feedbackward scenarios (in µs)")
pl.title("Evolution of execution time in function of the probility of sending message errors")
pl.show()