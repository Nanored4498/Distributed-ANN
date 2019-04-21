-module(network2).
-export([hidden_neurone_init/2]).

% Data structure for hidden neurones
-record(hd, {
	w, b, in, out, j,
	rcv_in, send_in, miss_rcv_in, miss_send_in, a=0,
	timeout_in=0, bit_in=0,
	rcv_out, send_out, miss_rcv_out, miss_send_out, d=0,
	timeout_out=0, bit_out=0
}).

hidden_neurone_init(Out, J) ->
	Size_Out = array:size(Out),
	receive In ->
		Size_In = array:size(In),
		W = array:map(fun(_I, _X) -> rand:normal() end, array:new(Size_In)),
		B = rand:normal(),
		Ain = array:new(Size_In, {default,0}),
		Aout = array:new(Size_Out, {default,0}),
		Data = #hd{w=W, b=B, in=In, out=Out, j=J,
					rcv_in=Ain, send_in=Aout,
					miss_rcv_in=Size_In, miss_send_in=0,
					rcv_out=Aout, send_out=Aout,
					miss_rcv_out=Size_Out, miss_send_out=0},
		hidden_neurone(Data)
	end.

hidden_neurone(Data0) ->
	Size_In = array:size(Data0#hd.in),
	Size_Out = array:size(Data0#hd.out),
	Bit_in = Data0#hd.bit_in,
	DT = 50000000,
	Time = os:system_time(),

	Data = Data0#hd{
		
		timeout_in = 
			if (Data0#hd.miss_send_in > 0), (Time - Data0#hd.timeout_in > DT) ->
				array:map(fun(K, V) ->
					if V == Bit_in ->
						WA = array:get(K, Data0#hd.w) * Data0#hd.a,
						array:get(K, Data0#hd.in) ! {forward, WA, Data0#hd.j, Bit_in};
					true ->
						ok
					end end,
					Data0#hd.send_in),
					Time;
			true ->
				Data0#hd.timeout_in
			end,

		timeout_out =
			if (Data0#hd.miss_send_out > 0), (Time - Data0#hd.timeout_out > DT) ->
				array:map(fun(K, V) ->
					if V == Data0#hd.bit_out ->
						array:get(K, Data0#hd.out) ! {backward, Data0#hd.d, Data0#hd.j, Data0#hd.bit_out};
					true ->
						ok
					end end,
					Data0#hd.send_out),
					Time;
			true ->
				Data0#hd.timeout_out
			end

	},

	receive

	{forward, ack, K, Bit_in} ->
		Send_K = array:get(K, Data#hd.send_in),
		if Send_K == Bit_in ->
			New_bit = (Bit_in + 1) rem 2,
			Missing2 = Data#hd.miss_send_in - 1,
			hidden_neurone(Data#hd{
				miss_send_in = Missing2,
				bit_in = if Missing2 == 0 -> New_bit; true -> Bit_in end,
				miss_rcv_in = if Missing2 == 0 -> Size_In; true -> 0 end
			});
		true ->
			hidden_neurone(Data)
		end;

	{forward, WA_K, K, Bit_in} ->
		%%% Send Ack %%%
		array:get(K, Data#hd.in) ! {forward, ack, Data#hd.j, Bit_in},
		%%%%%%%%%%%%%%%%
		Rcv_K = array:get(K, Data#hd.rcv_in),
		if Rcv_K == Bit_in ->
			New_bit = (Bit_in + 1) rem 2,
			Rcv_in2 = array:set(K, New_bit, Data#hd.rcv_in),
			A2 = if Data#hd.miss_rcv_in == Size_In -> WA_K;
					true -> Data#hd.a + WA_K end,
			Missing2 = Data#hd.miss_rcv_in - 1,
			%%% Executed when all inputs have been received %%%
			if Missing2 == 0 ->
				A3 = utils:sigmoid(A2 + Data#hd.b),
				array:map(fun(I, N) ->
					WI = array:gat(I, Data#hd.w),
					N ! {forward, WI*A3, Data#hd.j, Bit_in}
					end, Data#hd.out),
				hidden_neurone(Data#hd{
					rcv_in = Rcv_in2,
					miss_rcv_in = Missing2,
					miss_send_in = Size_Out,
					a = A3,
					timeout_in = os:system_time()
				});
			%%% Executed when some inputs haven't been already received %%%
			true ->
				hidden_neurone(Data#hd{
					rcv_in = Rcv_in2,
					miss_rcv_in = Missing2,
					a=A2
				})
			end;
		true ->
			hidden_neurone(Data)
		end

	after
		50 -> hidden_neurone(Data)
	end.