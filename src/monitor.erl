-module(monitor).
-export([launch/1, main/1, test/0]).

%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% TEST Function %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%

test() ->
	T = utils:read_csv("../data/training_set.csv"),
	M = monitor:launch([7, 5, 1]),
	io:format("======================================================~n"),
	io:format("================== test on an input ==================~n"),
	io:format("======================================================~n"),
	Input = [1, -1, 0.5, 2, -2, 0.5, 1],
	M ! {feedforward, array:from_list(Input), self()},
	receive X -> io:format("output of ~w is ~w~n", [Input, element(3, X)]) end,
	M ! {feedbackward, 0, self()},
	receive _ -> io:format("feed-backward to learn 0 done.~n") end,
	M ! {feedforward, array:from_list(Input), self()},
	receive X1 -> io:format("output of ~w is ~w~n", [Input, element(3, X1)]) end,
	io:format("======================================================~n"),
	io:format("================= backprop algorithm =================~n"),
	io:format("======================================================~n"),
	M ! {test, T, self()},
	receive X2 -> io:format("the mean quadratic error is: ~w~n", [element(3, X2)]) end,
	M ! {backprop, T, 4, self()},
	receive X3 -> io:format("=== ~w ===~n", [element(2, X3)]) end,
	M ! {test, T, self()},
	receive X4 -> io:format("the mean quadratic error is: ~w~n", [element(3, X4)]) end,
	M ! {backprop, T, 4, self()},
	receive X5 -> io:format("=== ~w ===~n", [element(2, X5)]) end,
	M ! {test, T, self()},
	receive X6 -> io:format("the mean quadratic error is: ~w~n", [element(3, X6)]) end,
	io:format("======================================================~n"),
	M ! shutdown,
	ok.

%%%%%%%%%%%%%%%%%%%%%%%%%

main(L) ->
	Net = network:neural_network(L, self()),
	[SI | _] = L,
	main2(Net, SI).

main2({I, O}=N, SI) ->
	receive

	{feedforward, Values, Pid} -> 
		Size_Vals = array:size(Values),
		if SI == Size_Vals ->
			I ! {forward, Values}, 
			receive {forward, Res} -> Pid ! {self(), feedforward, Res} end;
		true ->
			Pid ! {self(), err, "Bad input size"}
		end,
		main2(N, SI);

	{feedbackward, Y, Pid} ->
		O ! {backward, Y},
		receive {backward, Res} -> Pid ! {self(), feedbackward, Res} end,
		main2(N, SI);

	{backprop, TrainS, T, Pid} ->
		TrainS2 = get_valid_examples(TrainS, SI),
		backprop_algo(I, O, TrainS2, T, SI),
		Pid ! {self(), backprop_done},
		main2(N, SI);

	{test, TestS, Pid} ->
		TestS2 = get_valid_examples(TestS, SI),
		ST = array:size(TestS2),
		Errors = array:map(fun(_, X) ->
					I ! {forward, array:resize(SI, X)}, 
					Y = array:get(SI, X),
					receive {forward, A} -> math:pow(A - Y, 2) end
					end, TestS),
		Mean = lists:sum(array:to_list(Errors)) / ST / 2,
		Pid ! {self(), test, Mean},
		main2(N, SI);

	shutdown ->
		I ! shutdown

	end.

get_valid_examples(Set, SI) ->
	array:from_list(
		lists:filter(fun(X) -> SX = array:size(X)-1, SX == SI end,
			array:to_list(Set))).

backprop_algo(I, O, TrainS, T, SI) when T > 0 ->
	array:map(fun(_, X) ->
		I ! {forward, array:resize(SI, X)}, 
		receive {forward, _} -> ok end,
		O ! {backward, array:get(SI, X)},
		receive {backward, _} -> ok end
		end, TrainS),
	backprop_algo(I, O, TrainS, T-1, SI);
backprop_algo(_, _, _, _, _) -> ok.

launch(L) -> spawn(monitor, main, [L]).