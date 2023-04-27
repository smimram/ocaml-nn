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
let ( #- ) = M.sub

(** A network is a list of layers encoded as wji being the weight from the
    input i to the output j. *)
type t = Matrix.t list

(** Number of inputs of a nn. *)
let inputs (nn:t) = Matrix.cols (List.hd nn)

(** Create a networks with given number of links for each layer. *)
let create layers : t =
  let rec aux n = function
    | l::layers ->
      let w = Matrix.init l n (fun _ _ -> Random.float 2. -. 1.) in
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
  let rec aux nn yy =
    match nn, yy with
    | [], [y] ->
      (* Printf.printf "last y: %dx%d\n%!" (M.rows y) (M.cols y); *)
      let d = (y #- o) #* y #* (M.map (fun x -> 1. -. x) y) in
      (* Printf.printf "last d: %dx%d\n%!" (M.rows d) (M.cols d); *)
      d, []
    | w::nn, y::yy ->
      let d, ddy = aux nn yy in
      (* Printf.printf "y: %dx%d\n%!" (M.rows y) (M.cols y); *)
      (* Printf.printf "d: %dx%d\n%!" (M.rows d) (M.cols d); *)
      (* Printf.printf "w: %dx%d\n%!" (M.rows w) (M.cols w); *)
      let d = ((M.transpose w) #@ d) #* y #* (M.map (fun x -> 1. -. x) y) in
      (* Printf.printf "d': %dx%d\n%!" (M.rows d) (M.cols d); *)
      let dy = d #@ (M.transpose y) in
      d, dy::ddy
    | _ -> assert false
  in
  aux nn (propagate nn x) |> snd

(** Perform gradient descent (computed from [nn] and affecting [nn']). *)
let descent (nn:t) (nn':t) rate x o : t =
  List.map2 (fun w dy -> w #- (rate #. dy)) nn' (backpropagate nn x o)

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

let predict nn x =
  x
  |> M.of_list
  |> propagate nn
  |> List.last
  |> M.to_list
