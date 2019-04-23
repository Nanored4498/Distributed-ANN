all:
	erl -make

monitor: all
	erl -noshell -s monitor test -s init stop

network: all
	erl -noshell -s network test -s init stop

clean:
	rm *.beam