-module(testing).
-export([main/0]).

%%% List of networks that will be tested %%%
networks() ->
	[[7, 1],
	[7, 1, 1], [7, 4, 1], [7, 7, 1], [7, 10, 1],
	[7, 5, 5, 1], [7, 2, 8, 1], [7, 8, 2, 1],
	[7, 4, 4, 4, 1], [7, 7, 4, 1, 1], [7, 1, 4, 7, 1], [7, 3, 7, 3, 1]].

%%% Obtain the train set and the test set %%%
data() ->
	T = utils:read_csv("../data/training_set.csv"),
	ST = array:size(T),
	Lim = floor(ST*0.7),
	{L_train, L_test} = lists:split(Lim, array:to_list(T)),
	{array:from_list(L_train), array:from_list(L_test)}.

%%% Train a network N times %%%
step(_, _, _, 0) -> ok;
step(M, Train, Test, N) ->
	M ! {backprop, Train, 4, self()},
	receive {M, backprop_done} -> ok end,
	M ! {test, Test, self()},
	receive {M, test, Err} -> io:format("~w~n", [Err]) end,
	step(M, Train, Test, N-1).

%%% Print the net %%%
print_net([]) -> io:format("~n");
print_net([H | T]) -> io:format("~w ", [H]), print_net(T).

%%% Test a network %%%
test(Net, Train, Test) ->
	print_net(Net),
	M = monitor:launch(Net),
	Time = os:system_time(),
	step(M, Train, Test, 12),
	Time2 = os:system_time(),
	io:format("Time: ~w~n", [(Time2 - Time) div 1000]),
	M ! shutdown.

main() ->
	{Train, Test} = data(),
	lists:foreach(fun(Net) -> test(Net, Train, Test) end, networks()),
	io:format("END~n").