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

