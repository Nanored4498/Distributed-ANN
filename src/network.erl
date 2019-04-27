-module(network).
-export([hidden_neuron_init/3,
		input_neuron_init/2,
		output_neuron_init/1,
		neural_network/2, 
		test/0]).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% Time before resending a message in nanoseconds %%%%
dt() -> 20000000. %%%% 20ms
%%%% Learning rate %%%%
learning_rate() -> 0.6.
%%%% Verbose mode (print each message sent) %%%%
verbose() -> false.
%%%% Probability of failure when sending a message %%%%
err_prob() -> 0.0.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%% TESTING FUNCTION %%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% we want that the network outputs 1 on input (0.5, 0.4) %%%
learn(I, O) ->
	I ! {forward, array:from_list([0.5, 0.4])},
	receive {forward, Y2} -> io:format("res = ~w~n", [Y2]) end,
	O ! {backward, 1},
	receive {backward, Y3} -> io:format("res = ~w~n", [Y3]) end.

test() ->
	timer:sleep(100),
	T0 = os:system_time(),
	{I, O} = neural_network([2, 3, 1], self()),
	I ! {forward, array:from_list([0.2, 0.8])},
	O ! {forward, interruption},
	receive {forward, Y} -> io:format("res = ~w~n", [Y]) end, % Should be interruption as we sent an interruption
	learn(I, O),
	learn(I, O),
	learn(I, O),
	learn(I, O),
	learn(I, O),
	learn(I, O),
	learn(I, O),
	I ! {forward, array:from_list([0.5, 0.4])},
	receive {forward, Y6} -> io:format("res = ~w~n", [Y6]) end,
	O ! {backward, 1},
	I ! {backward, interruption},
	receive {backward, Y3} -> io:format("res = ~w~n", [Y3]) end, % Should be interruption as we sent an interruption
	T1 = os:system_time(),
	io:format("Time: ~w µs~n", [(T1 - T0) div 1000]),
	io:format("FINISH !!!!!~n"),
	done.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% SEND FUNCTION %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Send a message Msg to To
send(To, Msg) ->
	Err_Prob = err_prob(),
	R = rand:uniform(),
	V = verbose(),
	if (R > Err_Prob); (element(2, Msg) == interruption)->
		if V -> io:format("    ~w -> ~w: ~w~n", [self(), To, Msg]);
		true -> nothing end,
		To ! Msg;
	true -> 
		if V -> io:format("    ~w -> ~w: ~w <<fail>>~n", [self(), To, Msg]);
		true -> nothing end
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% HIDDEN neuron %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Data structure for neurons %%%
-record(com, {
	rcv,				% An array which store a bit for all neuron from which we will receive messages
	send,				% An array which store a bit for all neuron to which we will send messages
	miss_rcv,			% The number of messages that we need to receive
	miss_send=0,		% The number of ack that we need to receive
	timeout=0,			% The time in nano seconds of the last message sent
	bit=0,				% The bit that represent the state of the communication
	interrupted=false	% If true ignore received messages
}).
-record(hd, {
	w,				% w[i] is w_{ij}^{l+1} the weight of the connection with the i-th neuron of the out-layer
	b,				% bias
	in,				% in[k] is the PID of the k-th neuron of the in-layer
	out,			% out[i] is the PID of the i-th neuron of the out-layer
	j,				% indice of the neuron in its layer
	a=0,			% activation of the neuron
	d=0,			% error of the neuron
	input_layer		% true iff the neuron is in the input layer
}).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Initialization of the J-th neurone of a layer with out-layer Out.	%%%
%%% If the neuron is in the input layer then Input_Layer is true.		%%%
%%% Wait a message containing the in-layer								%%%
hidden_neuron_init(Out, J, Input_Layer) ->
	Size_Out = array:size(Out),
	receive In ->
		Size_In = array:size(In),
		W = array:map(fun(_I, _X) -> rand:normal() end, array:new(Size_Out)),
		B = if Input_Layer -> 0; true -> rand:normal() end,
		Ain = array:new(Size_In, {default,0}),
		Aout = array:new(Size_Out, {default,0}),
		Data = #hd{w=W, b=B, in=In, out=Out, j=J, input_layer=Input_Layer},
		For = #com{rcv=Ain, send=Aout, miss_rcv=Size_In},
		Back = #com{rcv=Aout, send=Ain, miss_rcv=Size_Out},
		hidden_neuron(Data, For, Back)
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Function to call when receiving a ack from I %%%
rcv_ack(Com, I) ->
	Send_I = array:get(I, Com#com.send),
	if Send_I == Com#com.bit ->
		New_bit = (Com#com.bit + 1) rem 2,
		Missing2 = Com#com.miss_send - 1,
		Send2 = array:set(I, New_bit, Com#com.send),
		Size = array:size(Com#com.rcv),
		if Missing2 == 0 ->
			Com#com{
				send = Send2,
				miss_send = Missing2,
				bit = New_bit,
				miss_rcv = Size
			};
		true ->
			Com#com{
				send = Send2,
				miss_send = Missing2
			}
		end;
	true ->
		Com
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Function to update Data when receiving the weighted value	%%%
%%% in-neuron K. First it true if it is the first value that we	%%%
%%% receive in this feed-forward scenario. Last is true if it	%%%
%%% is the last value that we need to receive. Bit is the one	%%%
%%% of the communication.										%%%
add_forward(Data, WA_K, First, Last, Bit) ->
	A2 = if First -> WA_K;
		true -> Data#hd.a + WA_K end,
	if Last ->
		A3 = utils:sigmoid(A2 + Data#hd.b),
		array:map(fun(I, N) ->
			WI = array:get(I, Data#hd.w),
			send(N, {forward, WI*A3, Data#hd.j, Bit})
			end, Data#hd.out),
		Data#hd{a=A3};
	true ->
		Data#hd{a=A2}
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Almost the same as the previous function expect that it is for	%%%
%%% feed-backward scenarion.										%%%
add_backward(Data, I, D_I, First, Last, Bit) ->
	W_I = array:get(I, Data#hd.w),
	D2 = if First -> W_I * D_I;
		true -> Data#hd.d + W_I * D_I end,
	Eta = learning_rate(),
	W2 = array:set(I, W_I - Eta * Data#hd.a * D_I, Data#hd.w),
	if Last ->
		D3 = D2 * utils:sigmoid_prime_of_sig(Data#hd.a),
		B2 = if Data#hd.input_layer -> 0;
				true -> Data#hd.b - Eta*D3 end,
		array:map(fun(_K, N) ->
			send(N, {backward, D3, Data#hd.j, Bit})
			end, Data#hd.in),
		Data#hd{d=D3, b=B2, w=W2};
	true ->
		Data#hd{d=D2, w=W2}
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Function called when receiving a value Val from the neuron I. It returns the couple	%%%
%%% (Data, Com) updated. Dir is forward if it is a message receive for the feed-forward	%%%
%%% scenario. Bit_rcv is the bit of the communication.									%%%
rcv_value(Data, Com, I, Val, Dir, Bit_rcv) ->
	if Com#com.interrupted ->
		{Data, Com};
	true ->
		Bit = Com#com.bit,
		%%% Send Ack %%%
		if (Bit_rcv == Bit); (Com#com.miss_send == 0) ->
			if Dir == forward ->
				send(array:get(I, Data#hd.in), {forward, ack, Data#hd.j, Bit_rcv});
			true ->
				send(array:get(I, Data#hd.out), {backward, ack, Data#hd.j, Bit_rcv})
			end;
		true ->
			this_is_a_message_of_a_new_wave_while_we_have_not_sent_our_message_to_all_neurons_of_next_layer
		end,
		%%%%%%%%%%%%%%%%
		Rcv_I = array:get(I, Com#com.rcv),
		if (Rcv_I == Bit), (Bit == Bit_rcv) ->
			New_bit = (Bit + 1) rem 2,
			Rcv2 = array:set(I, New_bit, Com#com.rcv),
			Size_Rcv = array:size(Com#com.rcv),
			Missing2 = Com#com.miss_rcv - 1,
			First = Com#com.miss_rcv == Size_Rcv,
			Last = Missing2 == 0,
			Data2 = if Dir == forward -> add_forward(Data, Val, First, Last, Bit);
					true -> add_backward(Data, I, Val, First, Last, Bit) end,
			%%% Executed when all inputs have been received %%%
			if Last ->
				{Data2, Com#com{
					rcv = Rcv2,
					miss_rcv = Missing2,
					miss_send = array:size(Com#com.send),
					timeout = os:system_time()
				}};
			%%% Executed when some inputs haven't been received yet %%%
			true ->
				{Data2, Com#com{
					rcv = Rcv2,
					miss_rcv = Missing2
				}}
			end;
		true ->
			{Data, Com}
		end
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Function called when receiving an interuption for feed-Dir scenario. The pass is 1 or 2	%%%
interrupt(Data, Com, Dir, Pass) ->
	if Data#hd.j == 0 ->		% The first neuron of the layer propagate the interruption
		if Dir == forward ->
			array:map(fun(_K, N) -> send(N, {forward, interruption, Pass}) end,
				if Pass == 1 -> Data#hd.in; true -> Data#hd.out end);
		true ->
			array:map(fun(_K, N) -> send(N, {backward, interruption, Pass}) end,
				if Pass == 1 -> Data#hd.out; true -> Data#hd.in end)
		end;
	true ->
		ok
	end,
	if Pass == 1 ->				% During the first pass only update the interrupted value to true
		Com#com{interrupted=true};
	true ->						% During the second pass reinitialize the record Com
		Size_Rcv = array:size(Com#com.rcv),
		Size_Send = array:size(Com#com.send),
		Com#com{
			rcv = array:new(Size_Rcv, {default,0}),
			send = array:new(Size_Send, {default,0}),
			miss_rcv = Size_Rcv,
			miss_send = 0,
			bit=0,
			interrupted=false
		}
	end.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Functions to resend datas %%%
resend_forward(Data, Com, I) ->
	WA = array:get(I, Data#hd.w) * Data#hd.a,
	send(array:get(I, Data#hd.out), {forward, WA, Data#hd.j, Com#com.bit}).

resend_backward(Data, Com, K) ->
	send(array:get(K, Data#hd.in), {backward, Data#hd.d, Data#hd.j, Com#com.bit}).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Function to resend data if necessary. Time is the current time %%%
time_com(Data, Com, Time, DT, Dir) -> Com#com{
	timeout = 
		if (Com#com.miss_send > 0), (Time - Com#com.timeout > DT) ->
			array:map(fun(I, V) ->
				if V == Com#com.bit ->
					if Dir == forward -> resend_forward(Data, Com, I);
					true -> resend_backward(Data, Com, I) end;
				true ->
					ok
				end end,
				Com#com.send),
			Time;
		true ->
			Com#com.timeout
		end
	}.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

hidden_neuron(Data, For0, Back0) ->
	Bit_for = For0#com.bit,
	Bit_back = Back0#com.bit,
	DT = dt(),
	Time = os:system_time(),

	For = time_com(Data, For0, Time, DT, forward),
	Back = time_com(Data, Back0, Time, DT, backward),

	receive

		%%%%%%%% ACK FORWARD %%%%%%%%
		{forward, ack, I, Bit_for} ->
			For2 = rcv_ack(For, I),
			hidden_neuron(Data, For2, Back);

		%%%%%%%% VALUE FORWARD %%%%%%%%
		{forward, WA_K, K, Bit_rcv} when WA_K /= ack ->
			{Data2, For2} = rcv_value(Data, For, K, WA_K, forward, Bit_rcv),
			hidden_neuron(Data2, For2, Back);

		%%%%%%%% INTERUPT FORWARD %%%%%%%%
		{forward, interruption, Pass} ->
			For2 = interrupt(Data, For, forward, Pass),
			hidden_neuron(Data, For2, Back);

		%%%%%%%% ACK BACKWARD %%%%%%%%
		{backward, ack, K, Bit_back} ->
			Back2 = rcv_ack(Back, K),
			hidden_neuron(Data, For, Back2);

		%%%%%%%% VALUE BACKWARD %%%%%%%%
		{backward, D_I, I, Bit_rcv} when D_I /= ack ->
			{Data2, Back2} = rcv_value(Data, Back, I, D_I, backward, Bit_rcv),
			hidden_neuron(Data2, For, Back2);

		%%%%%%%% INTERUPT BACKWARD %%%%%%%%
		{backward, interruption, Pass} ->
			Back2 = interrupt(Data, Back, backward, Pass),
			hidden_neuron(Data, For, Back2);

		%%%%%%%% SHUTDOWN %%%%%%%%
		shutdown ->
			if Data#hd.j == 0 ->
				array:map(fun(_I, N_I) -> N_I ! shutdown end, Data#hd.out);
			true -> ok end,
			V = verbose(),
			if V -> io:format("[~w] Shutdown !~n", [self()]);
			true -> ok end

	after
		DT div 1000000 -> hidden_neuron(Data, For, Back)
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% INPUT neuron %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-record(in, {
	monitor, layer, values,
	ack, miss=0, bit=0, timeout=0,
	ack_back, miss_back, bit_back=0
}).

input_neuron_init(Monitor, Layer) ->
	Size = array:size(Layer),
	Ack = array:new(Size, {default,0}),
	Data = #in{
		monitor = Monitor, layer = Layer, values = none,
		ack = Ack, ack_back = Ack, miss_back = Size
	},
	input_neuron(Data).

input_neuron(Data0) ->
	Bit = Data0#in.bit,
	Bit_back = Data0#in.bit_back,
	DT = dt(),
	Time = os:system_time(),

	Timeout2 =
		if (Data0#in.miss > 0), (Time - Data0#in.timeout > DT) ->
			array:map(fun(I, B) ->
				if B == Bit ->
					send(array:get(I, Data0#in.layer),
						{forward, array:get(I, Data0#in.values), 0, Bit});
				true ->
					ok
				end end, Data0#in.ack),
			Time;
		true -> Data0#in.timeout end,
	Data = Data0#in{timeout = Timeout2},

	receive

		{forward, interruption, 1} ->
			array:map(fun(_I, N_I) -> send(N_I, {forward, interruption, 2}) end, Data#in.layer),
			input_neuron(Data#in{
				bit = 0,
				ack = array:new(array:size(Data#in.layer), {default,0}),
				miss = 0
			});

		{forward, New_values}=Msg ->
			if Data#in.miss == 0 ->
				array:map(fun(I, V) ->
					send(array:get(I, Data#in.layer), {forward, V, 0, Bit})
				end, New_values),
				input_neuron(Data#in{
					values = New_values,
					miss = array:size(New_values),
					timeout = os:system_time()
				});
			true ->
				self() ! Msg,
				input_neuron(Data)
			end;

		{forward, ack, I, Bit} ->
			Ack_I = array:get(I, Data#in.ack),
			if Ack_I == Bit ->
				New_bit = (Bit + 1) rem 2,
				Miss2 = Data#in.miss - 1,
				Ack2 = array:set(I, New_bit, Data#in.ack),
				input_neuron(Data#in{
					ack = Ack2,
					miss = Miss2,
					bit = if Miss2 == 0 -> New_bit; true -> Bit end
				});
			true ->
				input_neuron(Data)
			end;

		{backward, _, I, Bit_rcv} ->
			send(array:get(I, Data#in.layer), {backward, ack, 0, Bit_rcv}),
			Ack_I = array:get(I, Data#in.ack_back),
			if (Ack_I == Bit_back), (Bit_back == Bit_rcv) ->
				New_bit = (Bit_back + 1) rem 2,
				Miss2 = Data#in.miss_back - 1,
				Ack2 = array:set(I, New_bit, Data#in.ack_back),
				if Miss2 == 0 -> 
					Data#in.monitor ! {backward, done},
					input_neuron(Data#in{
						ack_back = Ack2,
						miss_back = array:size(Data#in.ack_back),
						bit_back = New_bit
					});
				true ->
					input_neuron(Data#in{
						ack_back = Ack2,
						miss_back = Miss2
					})
				end;
			true ->
				input_neuron(Data)
			end;

		{backward, interruption} ->
			array:map(fun(_I, N_I) -> send(N_I, {backward, interruption, 1}) end, Data#in.layer),
			input_neuron(Data#in{miss_back=-1});
		
		{backward, interruption, 2} ->
			Data#in.monitor ! {backward, interruption},
			Size = array:size(Data#in.layer),
			input_neuron(Data#in{
				bit_back = 0,
				ack_back = array:new(Size, {default,0}),
				miss_back = Size
			});

		shutdown ->
			array:map(fun(_I, N_I) -> N_I ! shutdown end, Data#in.layer),
			V = verbose(),
			if V -> io:format("[~w] Shutdown !~n", [self()]);
			true -> ok end

	after DT div 1000000 -> input_neuron(Data)
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% OUTPUT neuron %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

-record(on, {
	monitor, layer, a=0, d=0, b,
	ack, miss=0, bit=0, timeout=0,
	ack_for, miss_for, bit_for=0
}).

output_neuron_init(Monitor) ->
	receive Layer ->
		Size = array:size(Layer),
		Ack = array:new(Size, {default,0}),
		Data = #on{
			monitor = Monitor, layer = Layer,
			b = rand:normal(),
			ack = Ack, ack_for = Ack, miss_for = Size
		},
		output_neuron(Data)
	end.

output_neuron(Data0) ->
	Bit = Data0#on.bit,
	Bit_for = Data0#on.bit_for,
	DT = dt(),
	Time = os:system_time(),

	Timeout2 =
		if (Data0#on.miss > 0), (Time - Data0#on.timeout > DT) ->
			array:map(fun(K, B) ->
				if B == Bit ->
					send(array:get(K, Data0#on.layer), {backward, Data0#on.d, 0, Bit});
				true ->
					ok
				end end, Data0#on.ack),
			Time;
		true -> Data0#on.timeout end,
	Data = Data0#on{timeout = Timeout2},

	receive

		{backward, interruption, 1} ->
			array:map(fun(_K, N_K) -> send(N_K, {backward, interruption, 2}) end, Data#on.layer),
			output_neuron(Data#on{
				bit = 0,
				ack = array:new(array:size(Data#on.layer), {default,0}),
				miss = 0
			});

		{backward, Y}=Msg ->
			if Data#on.miss == 0 ->
				D = (Data#on.a - Y) * utils:sigmoid_prime_of_sig(Data#on.a),
				array:map(fun(_K, N_K) ->
					send(N_K, {backward, D, 0, Bit})
				end, Data#on.layer),
				output_neuron(Data#on{
					d = D,
					b = Data#on.b - learning_rate()*D,
					miss = array:size(Data#on.layer),
					timeout = os:system_time()
				});
			true ->
				self() ! Msg,
				output_neuron(Data)
			end;

		{backward, ack, K, Bit} ->
			Ack_K = array:get(K, Data#on.ack),
			if Ack_K == Bit ->
				New_bit = (Bit + 1) rem 2,
				Miss2 = Data#on.miss - 1,
				Ack2 = array:set(K, New_bit, Data#on.ack),
				output_neuron(Data#on{
					ack = Ack2,
					miss = Miss2,
					bit = if Miss2 == 0 -> New_bit; true -> Bit end
				});
			true ->
				output_neuron(Data)
			end;

		{forward, WA_K, K, Bit_rcv} ->
			send(array:get(K, Data#on.layer), {forward, ack, 0, Bit_rcv}),
			Ack_K = array:get(K, Data#on.ack_for),
			if (Ack_K == Bit_for), (Bit_for == Bit_rcv) ->
				Size = array:size(Data#on.layer),
				A2 = if Data#on.miss_for == Size -> WA_K;
						true -> Data#on.a + WA_K end,
				New_bit = (Bit_for + 1) rem 2,
				Miss2 = Data#on.miss_for - 1,
				Ack2 = array:set(K, New_bit, Data#on.ack_for),
				if Miss2 == 0 ->
					A3 = utils:sigmoid(A2),
					Data#on.monitor ! {forward, A3},
					output_neuron(Data#on{
						ack_for = Ack2,
						miss_for = Size,
						bit_for = New_bit,
						a = A3
					});
				true ->
					output_neuron(Data#on{
						ack_for = Ack2,
						miss_for = Miss2,
						a = A2
					})
				end;
			true ->
				output_neuron(Data)
			end;

		{forward, interruption} ->
			array:map(fun(_K, N_K) -> send(N_K, {forward, interruption, 1}) end, Data#on.layer),
			output_neuron(Data#on{miss_for=-1});

		{forward, interruption, 2} ->
			Data#on.monitor ! {forward, interruption},
			Size = array:size(Data#on.layer),
			output_neuron(Data#on{
				bit_for = 0,
				ack_for = array:new(Size, {default,0}),
				miss_for = Size
			});

		shutdown ->
			V = verbose(),
			if V -> io:format("[~w] Shutdown !~n", [self()]);
			true -> ok end

	after DT div 1000000 -> output_neuron(Data)
	end.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% NETWORK CREATION %%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

neural_network([], _) ->
	erlang:error("The network has to have at least two layer");
neural_network(L, Monitor) ->
	[O|HL] = lists:reverse(L),
	if O == 1, HL /= [] ->
		Output = spawn(network, output_neuron_init, [Monitor]),
		Input = neural_network2(HL, array:from_list([Output]), Monitor),
		{Input, Output};
	true ->
		erlang:error("The network has to have only one output and need at least two layers.")
	end.

neural_network2([N], Out, Monitor) ->
	Layer = array:map(fun(J, _) -> spawn(network, hidden_neuron_init, [Out, J, true]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	Input = spawn(network, input_neuron_init, [Monitor, Layer]),
	In_array = array:from_list([Input]),
	array:map(fun(_I, Pid) -> Pid ! In_array end, Layer),
	Input;
neural_network2([N|L], Out, Monitor) ->
	Layer = array:map(fun(J, _) -> spawn(network, hidden_neuron_init, [Out, J, false]) end, array:new(N)),
	array:map(fun(_I, Pid) -> Pid ! Layer end, Out),
	neural_network2(L, Layer, Monitor).