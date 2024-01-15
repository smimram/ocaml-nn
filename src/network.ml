(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

module List = struct
  include List

  let rec last = function
    | [] -> assert false
    | [x] -> x
    | _::l -> last l
end

module M = Matrix
let ( #@ ) = M.mul
let ( #* ) = M.hadamard
let ( #. ) = M.cmul
let ( #+ ) = M.add
let ( #- ) = M.sub

(** A network is a list of layers encoded as w.(j).(i) being the weight from the
    input i to the output j. *)
type t = Matrix.t list

(** Number of inputs of a nn. *)
let inputs (nn:t) = M.src (List.hd nn)

(** Create a networks with given number of links for each layer. *)
let create layers : t =
  let rec aux n = function
    | l::layers ->
      let w = M.init l n (fun _ _ -> Random.float 2. -. 1.) in
      assert (M.src w = n && M.tgt w = l);
      w :: aux l layers
    | [] -> []
  in
  aux (List.hd layers) (List.tl layers)

(** Propagate input x along the network, keeping intermediate results. *)
let rec propagate (nn:t) x =
  match nn with
  | l::n -> x :: propagate n (M.map sigmoid (l #@ x))
  | [] -> [x]

(** Backpropagate to compute the δ. [x] is the input and [o] is the expected
    output. *)
let backpropagate (nn:t) x o =
  (* The function returns for each step the last δ and the list of products
     δy. *)
  let rec aux nn yy =
    match nn, yy with
    | [], [y] ->
      let d = (y #- o) #* y #* (M.map (fun x -> 1. -. x) y) in
      assert (M.src d = 1 && M.tgt d = M.tgt y);
      d, []
    | w::nn, y::yy ->
      let d, ddy = aux nn yy in
      assert (M.src d = 1);
      let d' = (M.transpose ((M.transpose d) #@ w)) #* y #* (M.map (fun x -> 1. -. x) y) in
      assert (M.src d = 1 && M.tgt d = M.tgt w);
      let dy = d #@ (M.transpose y) in
      assert (M.dims dy = M.dims w);
      d', dy::ddy
    | _ -> assert false
  in
  aux nn (propagate nn x) |> snd

(** Perform gradient descent (computed from [nn] and affecting [nn']). *)
let descent (nn:t) (nn':t) rate x o : t =
  List.map2 (fun w dy -> w #+ ((-.rate) #. dy)) nn' (backpropagate nn x o)

(** Compute mean error on given dataset. *)
let error nn data =
  List.map
    (fun (x,o) ->
       let y = List.last (propagate nn x) in
       let d = y #- o in
       M.to_scalar (M.transpose d #@ d)
    ) data
  |> List.fold_left (+.) 0.
  |> (fun x -> x /. float (List.length data))

(** Iterate backpropagation until given number of loops or precision have been
    reached. *)
let fit ?(layers=[]) ?(iterations=10_000) ?(precision=1e-3) ?(rate=0.2) data =
  let i, o =
    let x, y = List.hd data in
    List.length x, List.length y
  in
  let data = List.map (fun (x,o) -> M.of_list x, M.of_list o) data in
  let layers = i :: layers @ [o] in
  let nn = create layers in

  let rec loop i nn =
    if i >= iterations || error nn data <= precision then nn
    else
      let nn = List.fold_left (fun nn' (x,o) -> descent nn nn' rate x o) nn data in
      loop (i+1) nn
  in
  loop 0 nn

(** Predict the output on a given input. *)
let predict nn x =
  x
  |> M.of_list
  |> propagate nn
  |> List.last
  |> M.to_list
