(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

module List = struct
  include List

  let rec last = function
    | [] -> assert false
    | [x] -> x
    | _::l -> last l
end

module Int = struct
  include Int

  (** Fold a function from 0 to n-1. *)
  let fold f x n =
    let rec aux x i =
      if i >= n then x
      else aux (f x i) (i+1)
    in
    aux x 0
end

(** Matrices. As usual a.(i).(j) is the element in row i and column j and
    corresponds to the i-th output of the j-th input. *)
module Matrix = struct
  type t = float array array

  let init rows cols f : t =
    Array.init rows (fun i -> Array.init cols (fun j -> f i j))

  (** Number of rows. *)
  let rows (a:t) = Array.length a

  (** Number of columns. *)
  let cols (a:t) = Array.length a.(0)

  let to_scalar a =
    assert (rows a = 1);
    assert (cols a = 1);
    a.(0).(0)    

  let transpose a =
    init (cols a) (rows a) (fun i j -> a.(j).(i))

  (** Create a vector from a list. *)
  let of_list l =
    transpose [|Array.of_list l|]

  let to_list a =
    assert (cols a = 1);
    (transpose a).(0) |> Array.to_list
  
  (** Sum of f i for i from 0 to n - 1. *)
  let sum n f =
    let rec aux x i =
      if i >= n then x
      else aux (x +. f i) (i + 1)
    in
    aux 0. 0

  let map f (a:t) : t =
    Array.map (fun r -> Array.map (fun x -> f x) r) a

  let map2 f (a:t) (b:t) : t =
    assert (rows a = rows b);
    assert (cols a = cols b);
    Array.map2 (Array.map2 f) a b

  let add a b = map2 (+.) a b

  let cmul x a = map (fun y -> x *. y) a

  let neg = cmul (-1.)

  let sub a b = add a (neg b)

  let hadamard a b = map2 ( *. ) a b

  (** Matrix multiplication. *)
  let mul (a:t) (b:t) =
    assert (cols a = rows b);
    let n = cols a in
    init (rows a) (cols b) (fun i k -> Int.fold (fun x j -> x +. a.(i).(j) *. b.(j).(k)) 0. n)
end

module M = Matrix
let ( #@ ) = M.mul
let ( #* ) = M.hadamard
let ( #. ) = M.cmul
let ( #- ) = M.sub

module Network = struct
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
        let d = (y #- o) #* y #* (M.map (fun x -> 1. -. x) y) in
        d, []
      | l::nn, y::yy ->
        let d, ddy = aux nn yy in
        let d = M.transpose ((M.transpose d #@ l)) #* y #* (M.map (fun x -> 1. -. x) y) in
        let dy = d #* (M.transpose y) in
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
end
