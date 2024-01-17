(** Generic networks. *)

open Extlib
   
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
  let affine ?(regularization=0.) ~rate w b =
    assert (Matrix.tgt w = Vector.dim b);
    let inputs = Matrix.src w in
    let outputs = Matrix.tgt w in
    let w = ref w in
    let b = ref b in
    let forward x =
      assert (Vector.dim x = inputs);
      Vector.add (Matrix.app !w x) !b
    in
    let backward x g =
      assert (Vector.dim x = inputs);
      assert (Vector.dim g = outputs);
      let g' = Matrix.tapp !w g in
      let b' = Array.mapi (fun j b -> b -. rate *. (g.(j) +. regularization *. b)) !b in
      let w' = Matrix.mapi (fun j i w -> w -. rate *. (g.(j) *. x.(i) +. regularization *. w)) !w in
      w := w';
      b := b';
      g'
    in
    { inputs; outputs; forward; backward }
    
  (** Apply an activation function on each input. *)
  let activation kind n =
    let f, f' =
      match kind with
      | `Sigmoid -> (sigmoid, fun x -> let s = sigmoid x in s *. (1. -. s))
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
    let forward x = 0.5 *. Vector.squared_norm (Vector.sub x target) |> Vector.scalar in
    let backward x g =
      let x = Vector.sub x target in
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

  (** Apply the softmax function (to turn logits into probabilities). *)
  let softmax n =
    let forward x = Vector.softmax x in
    let backward _x _g =
      (* let s = forward x in *)
      (* TODO: check this *)
      failwith "TODO: check implementation.";
      (*
      Vector.init n
        (fun j ->
           let sj = s.(j) in
           Vector.mapi (fun i si -> g.(i) *. si *. (if i = j then (1. -. sj) else sj)) s |> Vector.sum
        )
         *)
    in
    {
      inputs = n;
      outputs = n;
      forward;
      backward
    }

  (** Softmax followed by cross-entropy. *)
  let softmax_ce n target =
    let forward x =
      let s = x |> Vector.map exp |> Vector.sum |> log in
      Vector.map (fun x -> s -. x) x
    in
    let backward x g =
      assert (g = Vector.scalar 1.);
      Vector.sub (Vector.softmax x) (Vector.init n (fun i -> if i = target then 1. else 0.))
    in
    {
      inputs = n;
      outputs = 1;
      forward;
      backward;
    }
end

open Layer

type t =
  {
    layers : Layer.t list
  }

let src (net:t) = (List.hd net.layers).inputs

let tgt (net:t) = (List.last net.layers).outputs

(** Create a network from a list of layers. *)
let make layers : t =
  assert (layers <> []);
  let n = (List.hd layers).inputs in
  let rec check n = function
    | layer::net -> assert (layer.inputs = n); check layer.outputs net
    | [] -> if n <> 1 then failwith "Expecting one output: did you forget to put an error measurement in the end?"
  in
  check n layers;
  { layers }

(** Append two networks. *)
let append net1 net2 =
  assert (tgt net1 = src net2);
  { layers = net1.layers@net2.layers }

(** Create a neural network with given arities for the layers and convergence
    rate. *)
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
  make (aux n layers)

(** Compute the output of a network on a given input. *)
let predict net x =
  let rec aux x = function
  | l::layers -> aux (l.forward x) layers
  | [] -> x
  in
  aux x net.layers

(** Forward propagation: returns layers decorated with their input, as well as
    the global output. *)
let forward net x =
  let rec aux x = function
    | [l] -> [x,l], l.forward x
    | l::layers ->
      let layers, o = aux (l.forward x) layers in
      (x,l)::layers, o
    | [] -> assert false
  in
  aux x net.layers

(** Backward propagation. Returns the previously computed error. *)
let backward net x =
  let layers, o = forward net x in
  let rec aux = function
    | (x,l)::net -> l.backward x (aux net)
    | [] -> Vector.scalar 1.
  in
  ignore (aux layers);
  Vector.to_scalar o

(** Train a network on given data. *)
let fit ?(distance=`Euclidean) net dataset =
  (* Distance layer. *)
  let distance target =
    match distance with
    | `Euclidean -> Layer.squared_distance target
  in
  let step x y =
    let distance = make [distance y] in
    let net = append net distance in
    (* List.iter (fun l -> Printf.printf "%d -> %d\n%!" (Layer.src l) (Layer.tgt l)) net; *)
    ignore (backward net x)
  in
  try List.iter (fun (x,y) -> step x y) dataset
  with Exit -> ()
