(* See https://nbviewer.org/gist/yoavg/d76121dfde2618422139 *)

module CharSet = Set.Make(struct type t = char let compare = compare end)

module StringMap = Map.Make (struct type t = string let compare = compare end)

module CharMap = struct
  include Map.Make(struct type t = char let compare = compare end)

  (* Map chars to the number of occurences. *)
  let of_list l =
    List.fold_left (fun m c -> update c (function Some n -> Some (n+1) | None -> Some 1) m) empty l

  (* Turn occurrences to probabilities. *)
  let proba m =
    let n = fold (fun _ i n -> n + i) m 0 in
    (* Printf.printf "n: %d\n%!" n; *)
    let n = float n in
    map (fun i -> float i /. n) m
end

let () =
  Random.self_init ();
  
  (* Read the file. *)
  (* let filename = "miserables.txt" in *)
  let filename = "shakespeare.txt" in
  let file =
    let ic = open_in_bin filename in
    let s = really_input_string ic (in_channel_length ic) in
    close_in ic;
    s
  in
  (* let file = file |> String.to_seq |> Seq.filter (fun c -> c <> '\r' && c <> '\n') |> String.of_seq in *)
  Printf.printf "Read file.\n%!";

  (* Find all characters. *)
  (*
  let cs = ref CharSet.empty in
  for i = 0 to String.length file - 1 do
    cs := CharSet.add file.[i] !cs
  done;
  let cs = CharSet.to_seq !cs |> List.of_seq |> List.sort compare in
  Printf.printf "%d characters: %s\n" (List.length cs) (List.to_seq cs |> String.of_seq);
  *)

  (* Compute n-grams. *)
  let order = 6 in
  let file = String.make order '~' ^ file in
  let ngram = ref StringMap.empty in
  for i = 0 to String.length file - order - 2 do
    let s = String.sub file i order in
    let c = file.[i+order] in
    ngram := StringMap.update s (function Some l -> Some (c::l) | None -> Some [c]) !ngram
  done;
  let ngram = StringMap.map (fun l -> l |> CharMap.of_list |> CharMap.proba) !ngram in

  (* Test. *)
  (*
  let test = StringMap.find "nobl" ngram |> CharMap.to_seq |> List.of_seq |> List.sort (fun (c,p) (c',p') -> compare (p,c) (p',c')) |> List.rev in
  Printf.printf "nobl\n%!";
  List.iter (fun (c,p) -> Printf.printf "%c (%d): %f\n%!" c (int_of_char c) p) test;
  *)

  (* Pick next letter for ngram. *)
  let pick s =
    let d = StringMap.find_opt s ngram in
    match d with
    | None -> char_of_int (int_of_char 'a' + Random.int 26)
    | Some d ->
      let rec aux last x d =
        match d () with
        | Seq.Cons ((c,p),d) ->
          if x < p then c else aux last (x -. p) d
        | Nil -> (* Printf.printf "Nil: %f\n%!" x; *) last
      in
      CharMap.to_seq d |> aux 'a' (Random.float 1.)
  in

  let s = ref (String.make order '~') in
  for _ = 0 to 5000 do
    let c = pick !s in
    print_char c;
    s := String.sub !s 1 (order - 1) ^ String.make 1 c
  done
