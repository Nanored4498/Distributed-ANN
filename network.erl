-module(network).
-export([neural_network/2, input_neurone/3, hidden_neurone_init/2]).

% TODO: Spawn on several nodes

input_neurone(Out, J, Wave) ->
	receive
	
	{backward, interruption, _}=Int ->
		if J == 0 ->
			array:map(fun(_Ind, N) -> N ! Int end, Out);
		true ->
			ok
		end,
		input_neurone(Out, J, Wave);

	{forward, X, Pid} ->
		Pid ! {wave_f, Wave},
		self() ! {forward, X},
		input_neurone(Out, J, Wave);

	{forward, X} ->
		array:map(fun(_Ind, N) -> N ! {forward, X, J, Wave} end, Out),
		input_neurone(Out, J, Wave+1)
	
	end.

% w : an array of weights. One value per input
% b : the bias (as a float)
% in : an array of PIDs of every inputs
% out : an array of PIDs of every outputs
% j : The index of the neurone in its layer
% in_vals : an array of the values received as input
% missing_in is the number of values from the inputs that are missing
% sum_in is the ponderated sum of all values in input
-record(hidden_data, {
	w, b, in, out, j,
	in_vals, missing_in, sum_in=0, wave_in=0, int_wave_in=sets:new(),
	out_vals, missing_out, sum_out=0, a_in=null, a=0, wave_out=0, int_wave_out=sets:new()
}).

next_wave(Wave, Int_Waves) ->
	Is_Interrupted = sets:is_element(Wave, Int_Waves),
	if Is_Interrupted  ->
		next_wave(Wave+1, sets:del_element(Wave, Int_Waves));
	true ->
		{Wave, Int_Waves}
	end.

% Also used as output neurone
% TODO: Manage the loss of messages
hidden_neurone_init(Out, J) ->
	receive In ->
		Size_In = array:size(In),
		Size_Out = array:size(Out),
		W = array:map(fun(_I, _X) -> rand:normal() end, array:new(Size_In)),
		Data = #hidden_data{w=W, b=rand:normal(), in=In, out=Out, j=J,
					in_vals=array:new(Size_In), missing_in=Size_In,
					out_vals=array:new(Size_Out), missing_out=Size_Out},
		hidden_neurone(Data)
	end.

hidden_neurone(Data) ->
	Size_In = array:size(Data#hidden_data.in),
	Size_Out = array:size(Data#hidden_data.out),
	Wave_In = Data#hidden_data.wave_in,
	Wave_Out = Data#hidden_data.wave_out,
	receive

	{forward, X_K, K, Wave_In} ->
		In_Vals_K = array:get(K, Data#hidden_data.in_vals),
		if In_Vals_K == undefined ->
			In_Vals2 = array:set(K, X_K, Data#hidden_data.in_vals),
			W_K = array:get(K, Data#hidden_data.w),
			Sum2 = Data#hidden_data.sum_in + X_K*W_K,
			Missing2 = Data#hidden_data.missing_in - 1,
			%%% Executed when all inputs have been received %%%
			if Missing2 == 0 ->
				Z = Sum2 + Data#hidden_data.b,
				A = utils:sigmoid(Z),
				array:map(fun(_I, N) -> N ! {forward, A, Data#hidden_data.j, Wave_In} end, Data#hidden_data.out),
				{Wave2, Int_Waves2} = next_wave(Wave_In+1, Data#hidden_data.int_wave_in),
				hidden_neurone(Data#hidden_data{
					in_vals=array:new(Size_In),
					missing_in=Size_In,
					sum_in=0,
					wave_in=Wave2,
					int_wave_in=Int_Waves2,
					a_in=In_Vals2,
					a=A
				});
			%%% Executed when some inputs haven't been already received %%%
			true ->
				hidden_neurone(Data#hidden_data{
					in_vals=In_Vals2,
					missing_in=Missing2,
					sum_in=Sum2
				})
			end;
		true ->
			hidden_neurone(Data)
		end;

	{forward, _, _, Wave}=M when Wave > Wave_In ->
		self() ! M,
		hidden_neurone(Data);

	{forward, interruption, Wave_In}=Int ->
		if Data#hidden_data.j == 0 ->
			array:map(fun(_K, N) -> N ! Int end, Data#hidden_data.in);
		true ->
			ok
		end,
		{Wave2, Int_Waves2} = next_wave(Wave_In+1, Data#hidden_data.int_wave_in),
		hidden_neurone(Data#hidden_data{
			in_vals=array:new(Size_In),
			missing_in=Size_In,
			sum_in=0,
			wave_in=Wave2,
			int_wave_in=Int_Waves2
		});

	{forward, interruption, Wave}=Int when Wave > Wave_In ->
		if Data#hidden_data.j == 0 ->
			array:map(fun(_K, N) -> N ! Int end, Data#hidden_data.in);
		true ->
			ok
		end,
		hidden_neurone(Data#hidden_data{
			int_wave_in=sets:add_element(Wave, Data#hidden_data.int_wave_in)
		});

	{forward, interruption, start, Wave} ->
		M = {forward, interruption, Wave},
		self() ! array:get(0, Data#hidden_data.out) ! M,
		hidden_neurone(Data);

	% {forward, resend, Wave, I} ->
	% 	M = {forward, Data#hidden_data.a, Data#hidden_data.j, Wave},
	% 	array:get(I, Data#hidden_data.in) ! M,
	% 	hidden_neurone(Data);

	{backward, D_I, I, Wave_Out} ->
		Out_Vals_I = array:get(I, Data#hidden_data.out_vals),
		if Out_Vals_I == undefined ->
			Sum2 = Data#hidden_data.sum_out + D_I,
			Missing2 = Data#hidden_data.missing_out - 1,
			if Missing2 == 0 ->
				D = Sum2 * utils:sigmoid_prime_of_sig(Data#hidden_data.a),
				%%%%%%%%%%%%%%%%%%%%% Send to previous layer %%%%%%%%%%%%%%%%%%%%%
				array:map(fun(K, N) ->
					N ! {backward, D * array:get(K, Data#hidden_data.w), Data#hidden_data.j, Wave_Out} end,
					Data#hidden_data.in),
				%%%%%%% Update with eta = 0.01 as proposed in the subject %%%%%%%%%
				W2 = array:map(fun(K, X) ->
					X - 0.01*array:get(K, Data#hidden_data.a_in)*D end,
					Data#hidden_data.w),
				B2 = Data#hidden_data.b - 0.01*D,
				%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				{Wave2, Int_Waves2} = next_wave(Wave_Out+1, Data#hidden_data.int_wave_out),
				hidden_neurone(Data#hidden_data{
					w=W2,
					b=B2,
					out_vals=array:new(Size_Out),
					missing_out=Size_Out,
					sum_out=0,
					wave_out=Wave2,
					int_wave_out=Int_Waves2
				});
			true ->
				Out_Vals2 = array:set(I, D_I, Data#hidden_data.out_vals),
				hidden_neurone(Data#hidden_data{
					out_vals=Out_Vals2,
					missing_out=Missing2,
					sum_out=Sum2
				})
			end;
		true ->
			hidden_neurone(Data)
		end;

	{backward, start, Y} ->
		self() ! {backward, Data#hidden_data.a - Y, 0, Wave_Out},
		hidden_neurone(Data);

	{backward, _, _, Wave}=M when Wave > Wave_Out ->
		self() ! M,
		hidden_neurone(Data);

	{backward, interruption, Wave_Out}=Int ->
		if Data#hidden_data.j == 0 ->
			array:map(fun(_I, N) -> N ! Int end, Data#hidden_data.out);
		true ->
			ok
		end,
		{Wave2, Int_Waves2} = next_wave(Wave_Out+1, Data#hidden_data.int_wave_out),
		hidden_neurone(Data#hidden_data{
			out_vals=array:new(Size_Out),
			missing_out=Size_Out,
			sum_out=0,
			wave_out=Wave2,
			int_wave_out=Int_Waves2
		});

	{backward, interruption, Wave}=Int when Wave > Wave_Out ->
		if Data#hidden_data.j == 0 ->
			array:map(fun(_I, N) -> N ! Int end, Data#hidden_data.out);
		true ->
			ok
		end,
		hidden_neurone(Data#hidden_data{
			int_wave_out=sets:add_element(Wave, Data#hidden_data.int_wave_out)
		})

	% {backward, resend, Wave, K} ->
	% 	M = {backward, Data#hidden_data.a, Data#hidden_data.j, Wave},
	% 	array:get(K, Data#hidden_data.out) ! M,
	% 	hidden_neurone(Data)

	after 100 ->

		if Size_In > Data#hidden_data.missing_in ->
			array:map(fun(K, V) ->
				if V == undefined ->
					array:get(K, Data#hidden_data.in) ! {forward, resend, Wave_In, Data#hidden_data.j};
				true ->
					ok
				end end,
				Data#hidden_data.in_vals);
		true ->
			ok
		end,
		if Size_Out > Data#hidden_data.missing_out ->
			array:map(fun(I, V) ->
				if V == undefined ->
					array:get(I, Data#hidden_data.out) ! {backward, resend, Wave_Out, Data#hidden_data.j};
				true ->
					ok
				end
				end, Data#hidden_data.out_vals
			);
		true ->
			ok
		end,
		hidden_neurone(Data)

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
	Layer = array:map(fun(J, _X) -> spawn(network, input_neurone, [Out, J, 0]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	Layer;
neural_network2([N|L], Out) ->
	Layer = array:map(fun(J, _X) -> spawn(network, hidden_neurone_init, [Out, J]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	neural_network2(L, Layer).