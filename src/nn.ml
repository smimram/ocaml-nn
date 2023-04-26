(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

(** Derivative of the sigmoid function. *)
let sigmoid' x = (sigmoid x) *. (1. -. sigmoid x)

module Matrix = struct
  type t = float array array

  let init rows cols f : t =
    Array.init rows (fun i -> Array.init cols (fun j -> f i j))

  let rows a = Array.length a

  let cols a = Array.length a.(0)

  (** Sum of f i for i from 0 to n - 1. *)
  let rec sum n f =
    let rec aux x i =
      if i >= n then x
      else aux (x +. f i) (i + 1)
    in
    aux 0. 0

  (** Matrix multiplication. *)
  let mult (a:t) (b:t) =
    assert (cols a = rows b);
    let n = cols a in
    init (rows a) (cols b)(fun i k -> sum n (fun j -> a.(i).(j) *. b.(j).(k)))     
end

let ( ** ) = Matrix.mult

module Network = struct
  (** A network is a list of layers encoded as wji being the weight from the
      input i to the output j. *)
  type t = Matrix.t list

  (** Number of inputs of a nn. *)
  let inputs (n:t) = Matrix.cols (List.hd n)

  (** Create a networks with given number of links for each layer. *)
  let create layers =
    let rec aux n = function
      | m::layers ->
        let w = Matrix.init m n (fun _ _ -> Random.float 2. -. 1.) in
        w :: aux m layers
      | [] -> []
    in
    aux (List.hd layers) (List.tl layers)

  (** Propagate input x along the network. *)
  let rec propagate (n:t) x =
    match n with
    | l::n -> x :: propagate n (Matrix.mult l x)
    | [] -> [x]
end
