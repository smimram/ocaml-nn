(** Generic networks. *)

open Extlib

(** Batch references. *)
module Batch = struct
  type t =
    {
      mutable contents : float;
      mutable batch : float list;
    }

  let make x =
    {
      contents = x;
      batch = []
    }

  let get r = r.contents

  let set r x = r.batch <- x::r.batch

  module Operations = struct
    let ref x = make x
    let (!) = get
    let (:=) = set
  end

  let collect r =
    let s, n = List.fold_left (fun (s,n) x -> s+.x, n+1) (0.,0) r.batch in
    r.contents <- s /. float n;
    r.batch <- []
end

(** Layers of a net. *)
module Layer = struct
  (** A layer. *)
  type t = {
    inputs : int;
    (** Number of inputs. *)
    outputs : int;
    (** Number of outputs. *)
    forward : Vector.t -> Vector.t;
    (** Forward computation. *)
    backward : Vector.t -> Vector.t -> Vector.t;
    (** Backward computation: given the input and the output gradient, update
        the weights and return the new gradient. *)
  }

  let src l = l.inputs

  let tgt l = l.outputs

  (** Each output is an affine combination of all the inputs. *)
  let affine ~rate w b =
    assert (Matrix.tgt w = Vector.dim b);
    let inputs = Matrix.src w in
    let outputs = Matrix.tgt w in
    let forward x =
      assert (Vector.dim x = inputs);
      Vector.add (Matrix.app w x) b
    in
    let backward x g =
      assert (Vector.dim x = inputs);
      assert (Vector.dim g = outputs);
      for j = 0 to Matrix.tgt w do
        b.(j) <- b.(j) -. rate *. g.(j);
        for i = 0 to Matrix.src w do
          w.(j).(i) <- g.(j) *. x.(i)
        done
      done;
      Matrix.app (Matrix.transpose w) g
    in
    { inputs; outputs; forward; backward }
    
  (** Apply an activation function on each input. *)
  let activation kind n =
    let f, f' =
      match kind with
      | `Sigmoid -> (sigmoid, fun x -> sigmoid x *. (1. -. sigmoid x))
      | `ReLU -> (relu, step)
    in
    let forward x = Vector.map f x in
    let backward x g =
      assert (Vector.dim x = n);
      assert (Vector.dim g = n);
      Array.map2 (fun g x -> f' x *. g) g x
    in
    {
      inputs = n;
      outputs = n;
      forward;
      backward;
    }

  (** Half of the square of the euclidean distance to a given target. *)
  let squared_distance target =
    let n = Vector.dim target in
    let forward x = 0.5 *. Vector.squared_norm (Vector.diff x target) |> Vector.scalar in
    let backward x g =
      let x = Vector.diff x target in
      let g = g.(0) in
      (* We usually start backpropagating from 1. *)
      if g = 1. then x else Vector.cmul g x
    in
    {
      inputs = n;
      outputs = 1;
      forward = forward;
      backward = backward   
    }
end

open Layer

type t = Layer.t list

let src (net:t) = (List.hd net).inputs

let tgt (net:t) = (List.last net).outputs

(** Create a network from a list of layers. *)
let make (net : Layer.t list) : t =
  assert (net <> []);
  let n = (List.hd net).inputs in
  let rec check n = function
    | layer::net -> assert (layer.inputs = n); check layer.outputs net
    | [] -> if n <> 1 then failwith "Expecting one output: did you forget to put an error measurement in the end?"
  in
  check n net;
  net

(** Create a neural network with given arities for the layers and convergence rate.. *)
let neural ?(activation=`Sigmoid) ~rate layers =
  assert (layers <> []);
  let n = List.hd layers in
  let layers = List.tl layers in
  let rec aux n = function
    | l::layers ->
      let w = Matrix.init l n (fun _ _ -> Random.float 2. -. 1.) in
      let b = Vector.init l (fun _ -> 0.) in
      [Layer.affine ~rate w b; Layer.activation activation l]@(aux l layers)
    | [] -> []
  in
  aux n layers

(** Compute the output of a network on a given input. *)
let rec predict (net:t) x =
  match net with
  | l::net -> predict net (l.forward x)
  | [] -> x

(** Forward propagation: returns layers decorated with their input, as well as
    the global output. *)
let forward (net:t) x =
  let rec aux x = function
    | [l] -> [x,l], l.forward x
    | l::net ->
      let net, o = aux (l.forward x) net in
      (x,l)::net, o
    | [] -> assert false
  in
  aux x net

(** Backward propagation. Returns the previously computed error. *)
let backward (net:t) x =
  let net, o = forward net x in
  let rec aux = function
    | (x,l)::net -> l.backward x (aux net)
    | [] -> Vector.scalar 1.
  in
  ignore (aux net);
  Vector.to_scalar o

(** Train a network on given data. *)
let fit ?(distance=`Euclidean) ?(precision=1e-3) net dataset =
  (* Distance layer. *)
  let distance target =
    match distance with
    | `Euclidean -> Layer.squared_distance target
  in
  let step x y =
    let net = net@[distance y] in
    List.iter (fun l -> Printf.printf "%d -> %d\n%!" (Layer.src l) (Layer.tgt l)) net;
    let o = backward net x in
    if o < precision then raise Exit
  in
  try List.iter (fun (x,y) -> step x y) dataset
  with Exit -> ()
