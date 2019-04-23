import pylab as pl

perfs = {}
times = {}

while True:
	line = input().split()
	if line[0] == "END": break
	net = tuple(map(int, line))
	perf = []
	while True:
		line = input().split()
		if line[0] == "Time:":
			times[net] = int(line[1])
			perfs[net] = perf
			break
		perf.append(float(line[0]))

style = ['-', '--', '-.']
i = 0
for n in perfs:
	y = perfs[n]
	x = [j+1 for j in range(len(y))]
	pl.plot(x, y, label=str(n), linestyle=style[i % 3])
	i += 1
pl.xlabel("Number of backprop realized")
pl.ylabel("Quadratic error on Test set")
pl.title("Evolution of quadratic error for different network architectures")
pl.legend()
pl.show()