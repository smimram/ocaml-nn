(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

(** Rectified linear unit. *)
let relu x = max 0. x

(** Step function. *)
let step x = if x <= 0. then 0. else 1.

module List = struct
  include List

  let rec map3 f l1 l2 l3 =
    match l1, l2, l3 with
    | x::l1, y::l2, z::l3 -> (f x y z)::(map3 f l1 l2 l3)
    | [], [], [] -> []
    | _ -> assert false

  let rec last = function
    | [] -> assert false
    | [x] -> x
    | _::l -> last l

  (** Shuffle the elements of a list. *)
  let shuffle l =
    l
    |> List.map (fun x -> Random.bits (), x)
    |> List.sort Stdlib.compare
    |> List.map snd
end
