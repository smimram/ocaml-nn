(** Generic networks. *)

open Extlib

(** A layer. *)
type layer = {
  inputs : int;
  (** Number of inputs. *)
  outputs : int;
  (** Number of outputs. *)
  forward : Vector.t -> Vector.t;
  (** Forward computation. *)
  backward : Vector.t -> Vector.t -> Vector.t;
  (** Backward computation: given the input and the output gradient, update the
      weights and return the new gradient. *)
}

type t = layer list

(** Create a network from a list of layers. *)
let make (net : layer list) : t =
  assert (net <> []);
  let n = (List.hd net).inputs in
  let rec check n = function
    | layer::net -> assert (layer.inputs = n); check layer.outputs net
    | [] -> if n <> 1 then failwith "Expecting one output: did you forget to put an error measurement in the end?"
  in
  check n net;
  net

(** Forward propagation: returns the output, as well as layers decorated with
    their input. *)
let forward (net:t) x =
  let rec aux x = function
    | [l] -> l.forward x, [x,l]
    | l::net ->
      let o, net = aux (l.forward x) net in
      o, (x,l)::net
    | [] -> assert false
  in
  aux x net

(** Backward propagation. *)
let backward (net:t) x =
  let _, net = forward net x in
  let rec aux g = function
    | (x,l)::net -> aux (l.backward x g) net
    | [] -> g
  in
  ignore (aux [|1.|] net)
  
(** Apply an activation function on each input. *)
let activation kind n =
  let f, f' =
    match kind with
    | `Sigmoid -> (sigmoid, fun x -> sigmoid x *. (1. -. sigmoid x))
    | `ReLU -> (relu, step)
  in
  {
    inputs = n;
    outputs = n;
    forward = Vector.map f;
    backward = fun x g -> Array.map2 (fun g x -> f' x *. g) g x
  }

(** Each output is an affine combination of all the inputs. *)
let affine ~rate w b =
  assert (Matrix.tgt w = Vector.dim b);
  let inputs = Matrix.src w in
  let outputs = Matrix.tgt w in
  let forward x = Vector.add (Matrix.app w x) b in
  let backward x g =
    for j = 0 to Matrix.tgt w do
      b.(j) <- b.(j) -. rate *. g.(j);
      for i = 0 to Matrix.src w do
        w.(j).(i) <- g.(j) *. x.(i)
      done
    done;
    Matrix.app (Matrix.transpose w) g
  in
  { inputs; outputs; forward; backward }

(** Half of the square of the euclidean distance to a given target. *)
let square_distance target =
  let n = Vector.dim target in
  let backward x g =
    let g = g.(0) in
    if g = 1. then x else Vector.cmul g x
  in
  {
    inputs = n;
    outputs = 1;
    forward = Vector.hadamard target;
    backward = backward   
  }

