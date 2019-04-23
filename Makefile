all:
	cd src; erl -make

monitor: all
	cd src; erl -noshell -s monitor test -s init stop

network: all
	cd src; erl -noshell -s network test -s init stop

testing: all
	cd src; erl -noshell -s testing main -s init stop

clean:
	rm *.beam