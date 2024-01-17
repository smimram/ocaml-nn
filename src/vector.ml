type t = float array

let dim (x:t) = Array.length x

let scalar x : t = [|x|]

let to_scalar (x:t) =
  assert (dim x = 1);
  x.(0)

let map f (x:t) : t = Array.map f x

let map2 f (x:t) (y:t) : t = Array.map2 f x y

let add (x:t) (y:t) : t = Array.map2 (+.) x y

let diff (x:t) (y:t) : t = Array.map2 (-.) x y

let squared_norm (x:t) = Array.fold_left (fun s x -> s +. x *. x) 0. x

let cmul a x = map (fun x -> a *. x) x

let hadamard x y = map2 ( *. ) x y

let init n f : t = Array.init n f
