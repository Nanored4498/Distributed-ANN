-module(monitor).
-export([launch/1, main/1]).

main(L) ->
	Net = network:neural_network(L, self()),
	main2(Net).
main2({Input, _}=N) ->
	receive

	{feedforward, In, Pid} -> 
		Size_In = array:size(In),
		Size_Input = array:size(Input),
		if Size_In == Size_Input ->
			feedforward_algo(N, In, fun(Y) -> Pid ! {self(), feedforward, Y} end);
		true ->
			Pid ! {self(), err, "Bad input size"}
		end;

	{feedbackward, Y, Pid} ->
		feedbackward_algo(N, Y),
		Pid ! {self(), feedbackward_done};

	{backprop, TrainS, T, Pid} ->
		backprop_algo(N, TrainS, T),
		Pid ! {self(), backprop_done};

	{test, TestS, Pid} ->
		SI = array:size(Input),
		ST = array:size(TestS),
		Errors = array:map(fun(_, X_I) ->
			SX = array:size(X_I),
			if SX-1 == SI ->
				feedforward_algo(N, X_I, fun(A) -> math:pow(A - array:get(SI, X_I), 2) end);
			true ->
				1
			end end,
			TestS),
		Mean = array:foldl(fun(_I, X, Sum) -> X+Sum end, 0, Errors) / ST / 2,
		Pid ! {self(), ok, Mean}

	end, main2(N).

backprop_algo(_, _, 0) -> ok;
backprop_algo({Input, _}=N, TrainS, T) ->
	SI = array:size(Input),
	array:map(fun(_, X_I) ->
		SX = array:size(X_I),
		if SX-1 == SI ->
			feedforward_algo(N, X_I, fun(_) -> feedbackward_algo(N, array:get(SI, X_I)) end);
		true ->
			bad_line_size
		end end,
		TrainS),
	backprop_algo(N, TrainS, T-1).

feedforward_algo({Input, _}, In, Fun) ->
	array:map(fun(J, N_J) ->
		if J == 0 -> N_J ! {forward, array:get(J, In), self()};
		true -> N_J ! {forward, array:get(J, In)}
		end end,
		Input),
	wait_wave(Fun).
wait_wave(Fun) ->
	receive
		{wave_f, Wf} ->	wait_res(Wf, Fun);
		M -> self() ! M, wait_wave(Fun)
	end.
wait_res(Wf, Fun) ->
	receive
		{forward, Y, 0, Wf} -> Fun(Y);
		M -> self() ! M, wait_res(Wf, Fun)
	end.

feedbackward_algo({_, Output}, Y) ->
	Output ! {backward, start, Y}.

% interrupt_forward({_, Out}, Wave) ->
% 	Out ! {forward, interruption, start, Wave},
% 	receive {forward, interruption, Wave} -> ok0 after 100 -> ok1 end.

% interrupt_backward({In, _}, Wave) ->
% 	array:map(fun(_I, Pid) -> Pid ! {backward, interruption, Wave} end, In),
% 	receive {backward, interruption, Wave} -> ok0 after 1000 -> ok1 end.

launch(L) -> spawn(monitor, main, [L]).

% c(network).
% c(monitor).
% c(utils).
% T = utils:read_csv("training_set.csv"), ok.
% M = monitor:launch([7, 10, 10, 1]).
% M ! {feedforward, array:from_list([1, -1, 0.5, 2, -2, 0.5, 1]), self()}.
% M ! {test, T, self()}, ok.
% M ! {backprop, T, 1, self()}, ok.
% M ! {test, T, self()}, ok.
% flush().