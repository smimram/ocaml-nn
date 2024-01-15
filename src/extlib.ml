(** The sigmoid activation function. *)
let sigmoid x = 1. /. (1. +. (exp (-.x)))

(** Rectified linear unit. *)
let relu x = max 0. x

module List = struct
  include List

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
