import pylab as pl

perfs = {}
times = {}

while True:
	line = input().split()
	if line[0] == "END": break
	if ord(line[0][0]) > ord('9') or ord(line[0][0]) < ord('0'): continue
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

def numb_param(net):
	res = net[0]
	for i in range(len(net)-1):
		res += net[i+1]*(1 + net[i])
	return res

tns = sorted([(numb_param(net), times[net], net) for net in times])
ps = [a[0]*10**5 for a in tns]
ts = [a[1] for a in tns]
ns = [str(a[2]) for a in tns]
x = list(range(len(ts)))
pl.bar(x, ts, alpha=0.8)
pl.scatter(x, ps)

pl.xticks(x, ns)
pl.xlabel("architectures")
pl.ylabel("time of execution (in µs)")
pl.title("Time of execution for different architectures")
pl.show()