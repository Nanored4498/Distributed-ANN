% Collection of utils for reading/writing csv files
% based on reply https://stackoverflow.com/questions/1532081/csv-parser-in-erlang
% TODO: make distributed ?
-module(utils).
-export([read_csv/1,
		sigmoid/1, sigmoid_prime/1, sigmoid_prime_of_sig/1,
		 tanh/1, tanh_prime/1,
		 activate/2, derivative/2]).


% Activation functions
sigmoid(X) -> 1 / (1 + math:exp(-X)).
sigmoid_prime(X) -> sigmoid(X) * (1 - sigmoid(X)).
sigmoid_prime_of_sig(X) -> X * (1 - X).

tanh(X) -> 2 * sigmoid(2 * X) - 1.
tanh_prime(X) -> 1 - tanh(X) * tanh(X).

activate(Value, sigmoid) -> sigmoid(Value);
activate(Value, tanh) -> tanh(Value).

derivative(Value, sigmoid) -> sigmoid_prime(Value);
derivative(Value, tanh) -> tanh_prime(Value).


% Read csv into a dataframe with floats
read_csv(File) ->
	OpenRes = file:open(File, [read, raw]),
	case OpenRes of
		{ok, F} -> 
			parse_csv(F, file:read_line(F), []);
		_ -> io:format("Cannot open file '~s'~n", [File])
	end.

parse_csv(F, eof, Result) ->
	file:close(F),
	array:from_list(lists:reverse(Result));    
parse_csv(F, {ok, Line}, Result) ->
	parse_csv(F, file:read_line(F), [parse_line(Line)|Result]).


parse_line(Line) -> parse_line(Line, []).

parse_line([], Fields) ->
	array:from_list(lists:map(
		fun(X) -> 
			{FloatEl, _} = string:to_float(X),
			if FloatEl == error ->
				{IntEl, _} = string:to_integer(X),
				IntEl;
			true ->
				FloatEl
			end
		end,
		lists:reverse(Fields)));
parse_line("," ++ Line, Fields) -> parse_field(Line, Fields);
parse_line(Line, Fields) ->
	parse_field(Line, Fields).

parse_field("\"" ++ Line, Fields) -> parse_field_q(Line, Fields);
parse_field(Line, Fields) -> parse_field(Line, [], Fields).

parse_field("," ++ _ = Line, Buf, Fields) -> parse_line(Line, [lists:reverse(Buf)|Fields]);
parse_field([C|Line], Buf, Fields) -> parse_field(Line, [C|Buf], Fields);
parse_field([], Buf, Fields) -> parse_line([], [lists:reverse(Buf)|Fields]).

parse_field_q(Line, Fields) -> parse_field_q(Line, [], Fields).
parse_field_q("\"\"" ++ Line, Buf, Fields) -> parse_field_q(Line, [$"|Buf], Fields);
parse_field_q("\"" ++ Line, Buf, Fields) -> parse_line(Line, [lists:reverse(Buf)|Fields]);
parse_field_q([C|Line], Buf, Fields) -> parse_field_q(Line, [C|Buf], Fields).