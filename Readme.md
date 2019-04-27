# Distributed Neural Network

This an implementation of a neural network where each neuron is a single process. There is a monitor that allow you to spawn neural networks and launch bakprop alogrithm or test the efficiency of the net on a test set. To learn to use the monitor you can look at the function `test` in the file [src/monitor.erl](src/monitor.erl).  
There are also some parameters that can be changed at the beggining of the file [src/network.erl](src/network.erl). To test the code you can use:
```
# To test the network on simple feed-forward and feed-backward scenarios
make network
# To test the use of the monitor on data-sets
make monitor
# Finally to make some stats on different network architectures
make testing | python3 testing/stats.py
```

For more details there is a report in the report folder. It is a LaTeX file so you will need to compile it with `pdflatex report.tex` for example.