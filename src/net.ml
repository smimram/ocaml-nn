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

(** Apply an activation function on each input. *)
let activation kind n : layer =
  let f, f' =
    match kind with
    | `Sigmoid -> (sigmoid, fun x -> sigmoid x *. (1. -. sigmoid x))
  in
  {
    inputs = n;
    outputs = n;
    forward = Array.map f;
    backward = fun x g -> Array.map2 (fun g x -> g *. f' x) g x
  }

(** Each output is an affine combination of all the inputs. *)
let affine w b =
  assert (Matrix.tgt w = Vector.dim b);
  let inputs = Matrix.src w in
  let outputs = Matrix.tgt w in
  let forward x = Vector.add (Matrix.app w x) b in
  let backward _x g = Matrix.app (Matrix.transpose w) g in
  { inputs; outputs; forward; backward }
