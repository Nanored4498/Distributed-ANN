-module(network).
-export([neural_network/2, input_neurone/2, hidden_neurone_init/2]).

% TODO: Spawn on several nodes

input_neurone(Out, J) ->
	% io:format("Input neurone ~p~n", [J]),
	receive
	
	{backward, interruption, _}=Int ->
		if J == 0 ->
			array:map(fun(_Ind, N) -> N ! Int end, Out);
		true ->
			ok
		end,
		input_neurone(Out, J);

	{forward, X} ->
		array:map(fun(_Ind, N) -> N ! {forward, X, J} end, Out),
		input_neurone(Out, J)
	
	end.

% w: an array of weights. One value per input
% b: the bias (as a float)
% in: an array of PIDs of every inputs
% out: an array of PIDs of every outputs
% j: The index of the neurone in its layer
% in_vals: an array of the values received as input
% missing_in: the number of values from the inputs that are missing
% sum_in: the ponderated sum of all values in input
% missing_ack_in: the number of ack from the outputs that are missing
-record(hd, {
	w, b, in, out, j,
	in_vals, missing_in, sum_in=0,
	timeout_in=0,
	out_vals, missing_out, sum_out=0, a_in=null, a=0
}).

% Also used as output neurone
% TODO: Manage the loss of messages
hidden_neurone_init(Out, J) ->
	receive In ->
		Size_In = array:size(In),
		Size_Out = array:size(Out),
		W = array:map(fun(_I, _X) -> rand:normal() end, array:new(Size_In)),
		Data = #hd{w=W, b=rand:normal(), in=In, out=Out, j=J,
					in_vals=array:new(Size_In), missing_in=Size_In,
					out_vals=array:new(Size_Out), missing_out=Size_Out},
		hidden_neurone(Data)
	end.

hidden_neurone(Data) ->
	% io:format("Hidden neurone ~p~n", [Data#hd.j]),
	Size_In = array:size(Data#hd.in),
	Size_Out = array:size(Data#hd.out),

	Time = os:system_time(),
	if (Size_In > Data#hd.missing_in), (Time - Data#hd.timeout_in > 50000000) ->
		array:map(fun(K, V) ->
			if V == undefined ->
				array:get(K, Data#hd.in) ! {forward, resend, Data#hd.j};
			true ->
				ok
			end end,
			Data#hd.in_vals);
	true ->
		ok
	end,
	% if Size_Out > Data#hd.missing_out ->
	% 	array:map(fun(I, V) ->
	% 		if V == undefined ->
	% 			array:get(I, Data#hd.out) ! {backward, resend, Data#hd.j};
	% 		true ->
	% 			ok
	% 		end
	% 		end, Data#hd.out_vals
	% 	);
	% true ->
	% 	ok
	% end,
	% hidden_neurone(Data)

	receive

	{forward, X_K, K} ->
		In_Vals_K = array:get(K, Data#hd.in_vals),
		if In_Vals_K == undefined ->
			In_Vals2 = array:set(K, X_K, Data#hd.in_vals),
			W_K = array:get(K, Data#hd.w),
			Sum2 = Data#hd.sum_in + X_K*W_K,
			Missing2 = Data#hd.missing_in - 1,
			%%% Executed when all inputs have been received %%%
			if Missing2 == 0 ->
				Z = Sum2 + Data#hd.b,
				A = utils:sigmoid(Z),
				array:map(fun(_I, N) -> N ! {forward, A, Data#hd.j} end, Data#hd.out),
				hidden_neurone(Data#hd{
					in_vals=array:new(Size_In),
					missing_in=Size_In,
					sum_in=0,
					a_in=In_Vals2,
					a=A
				});
			%%% Executed when some inputs haven't been already received %%%
			true ->
				hidden_neurone(Data#hd{
					in_vals=In_Vals2,
					missing_in=Missing2,
					sum_in=Sum2,
					timeout_in = os:system_time()
				})
			end;
		true ->
			hidden_neurone(Data)
		end;

	{forward, interruption}=Int ->
		if Data#hd.j == 0 ->
			array:map(fun(_K, N) -> N ! Int end, Data#hd.in);
		true ->
			ok
		end,
		hidden_neurone(Data#hd{
			in_vals=array:new(Size_In),
			missing_in=Size_In,
			sum_in=0
		});

	% {forward, resend, Wave, I} ->
	% 	M = {forward, Data#hd.a, Data#hd.j, Wave},
	% 	array:get(I, Data#hd.in) ! M,
	% 	hidden_neurone(Data);

	{backward, start, Y} ->
		self() ! {backward, Data#hd.a - Y, 0},
		hidden_neurone(Data);

	{backward, D_I, I} ->
		Out_Vals_I = array:get(I, Data#hd.out_vals),
		if Out_Vals_I == undefined ->
			Sum2 = Data#hd.sum_out + D_I,
			Missing2 = Data#hd.missing_out - 1,
			if Missing2 == 0 ->
				D = Sum2 * utils:sigmoid_prime_of_sig(Data#hd.a),
				%%%%%%%%%%%%%%%%%%%%% Send to previous layer %%%%%%%%%%%%%%%%%%%%%
				array:map(fun(K, N) ->
					N ! {backward, D * array:get(K, Data#hd.w), Data#hd.j} end,
					Data#hd.in),
				%%%%%%% Update with eta = 0.01 as proposed in the subject %%%%%%%%%
				W2 = array:map(fun(K, X) ->
					X - 0.01*array:get(K, Data#hd.a_in)*D end,
					Data#hd.w),
				B2 = Data#hd.b - 0.01*D,
				%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				hidden_neurone(Data#hd{
					w=W2,
					b=B2,
					out_vals=array:new(Size_Out),
					missing_out=Size_Out,
					sum_out=0
				});
			true ->
				Out_Vals2 = array:set(I, D_I, Data#hd.out_vals),
				hidden_neurone(Data#hd{
					out_vals=Out_Vals2,
					missing_out=Missing2,
					sum_out=Sum2
				})
			end;
		true ->
			hidden_neurone(Data)
		end;

	{backward, interruption}=Int ->
		if Data#hd.j == 0 ->
			array:map(fun(_I, N) -> N ! Int end, Data#hd.out);
		true ->
			ok
		end,
		hidden_neurone(Data#hd{
			out_vals=array:new(Size_Out),
			missing_out=Size_Out,
			sum_out=0
		})

	% {backward, resend, Wave, K} ->
	% 	M = {backward, Data#hd.a, Data#hd.j, Wave},
	% 	array:get(K, Data#hd.out) ! M,
	% 	hidden_neurone(Data)

	after
		50 -> hidden_neurone(Data)
	end.

neural_network([], _) ->
	erlang:error("The network has to have at least one layer");
neural_network(L, Send_To) ->
	[O|HL] = lists:reverse(L),
	if O == 1, HL /= [] ->
		Output = spawn(network, hidden_neurone_init, [array:from_list([Send_To]), 0]),
		Input = neural_network2(HL, array:from_list([Output])),
		{Input, Output};
	true ->
		erlang:error("The network has to have only one output and need at least two layers.")
	end.

neural_network2([N], Out) ->
	Layer = array:map(fun(J, _X) -> spawn(network, input_neurone, [Out, J]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	Layer;
neural_network2([N|L], Out) ->
	Layer = array:map(fun(J, _X) -> spawn(network, hidden_neurone_init, [Out, J]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	neural_network2(L, Layer).