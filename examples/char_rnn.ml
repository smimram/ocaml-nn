(* See
   http://karpathy.github.io/2015/05/21/rnn-effectiveness/
   https://github.com/karpathy/char-rnn
*)

module CharSet = Set.Make(struct type t = char let compare = compare end)

let () =
  (* Read the file. *)
  let filename = "shakespeare.txt" in
  let file =
    let ic = open_in_bin filename in
    let s = really_input_string ic (in_channel_length ic) in
    close_in ic;
    s
  in

  (* Find all characters. *)
  let cs = ref CharSet.empty in
  for i = 0 to String.length file - 1 do
    let c = file.[i] in
    if c <> '\r' && c <> '\n' then
      cs := CharSet.add file.[i] !cs
  done;
  let cs = CharSet.to_seq !cs |> List.of_seq |> List.sort compare in
  Printf.printf "%d characters: %s\n" (List.length cs) (List.to_seq cs |> String.of_seq)
